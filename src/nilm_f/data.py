"""Chronological, training-only preprocessing for household power data.

Valid 576-point windows are enumerated before selecting every 576th valid
end. Missing rows invalidate windows without compressing the time axis."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from .loss import prepare_aux


WINDOW = 576
PROTOCOL = "train_only_v3"
DEFAULT_SCOPE = Path(__file__).resolve().parent / "resources" / "data_scope.json"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def array_hash(value):
    """Match the original hash, including dtype and shape as well as bytes."""
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        allow_nan=False,
    ).encode()).hexdigest()


def scaler_manifest(scaler):
    return {key: np.asarray(getattr(scaler, key + "_")).tolist()
            for key in ("mean", "scale", "var", "n_samples_seen")}


class WindowDataset(torch.utils.data.Dataset):
    """CPU tensors with shape [number of windows, 576]."""

    def __init__(self, x, y):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class IndexedWindows(torch.utils.data.Dataset):
    def __init__(self, dataset, stop):
        if not 0 < stop <= len(dataset):
            raise ValueError("Invalid fit window count")
        self.dataset, self.start, self.stop = dataset, 0, stop

    def __len__(self):
        return self.stop

    def __getitem__(self, index):
        x, y = self.dataset[index]
        return x, y, index


def split_boundaries(n_rows):
    """Use raw row counts, never held-out values or missingness."""
    outer = int(0.8 * n_rows)
    nominal_windows = max(0, (outer - 1) // WINDOW)
    fit_windows = int(0.9 * nominal_windows)
    return outer, fit_windows * WINDOW


def window_starts(values, begin, end, *, omit_final=True):
    """Return valid-end downselected starts in [begin, end) source rows."""
    segment = values[begin:end]
    invalid = ~np.isfinite(segment).all(axis=1)
    cumulative = np.concatenate(([0], np.cumsum(invalid, dtype=np.int64)))
    ends = np.arange(WINDOW, len(segment) + (not omit_final), dtype=np.int64)
    valid = ends[(cumulative[ends] - cumulative[ends - WINDOW]) == 0]
    return valid[::WINDOW] - WINDOW + begin


def _windows(normalized, starts):
    x = np.empty((len(starts), WINDOW), dtype=np.float32)
    y = np.empty_like(x)
    for offset in range(0, len(starts), 1024):
        rows = starts[offset:offset + 1024, None] + np.arange(WINDOW)[None, :]
        x[offset:offset + 1024] = normalized[rows, 0]
        y[offset:offset + 1024] = normalized[rows, 1]
    return x, y


def _support_manifest(starts, raw_interval):
    return {
        "allowed_raw_row_interval": list(raw_interval),
        "windows": len(starts), "points": len(starts) * WINDOW,
        "first_start_inclusive": int(starts[0]) if len(starts) else None,
        "last_end_exclusive": int(starts[-1] + WINDOW) if len(starts) else None,
        "starts_sha256": array_hash(starts),
        "nonoverlapping": bool(not len(starts) or np.all(np.diff(starts) >= WINDOW)),
    }


def prepare_phase(data, phase, smoke=False):
    """Attach fit/validation loaders and training-only Huber statistics.

    The optional smoke flag is retained for call-site compatibility. Window
    caps must have been applied by dataframe_to_client; this function never
    changes a split or silently truncates data. Repeated calls are safe.
    """
    if phase not in ("development", "formal"):
        raise ValueError("Invalid phase")
    for item in data:
        if item["manifest"]["preprocessing"]["phase"] != phase:
            raise ValueError("Phase-specific scalers cannot be reused across phases")
        original = item["train_dataset"]
        count = item["manifest"]["fit_windows"]
        item["fit_dataset"] = IndexedWindows(original, count)
        item["fit_loader"] = torch.utils.data.DataLoader(
            item["fit_dataset"], batch_size=64, shuffle=True)
        if phase == "development":
            if count >= len(original):
                raise ValueError("Insufficient training windows for internal validation")
            item["validation_loader"] = torch.utils.data.DataLoader(
                torch.utils.data.Subset(original, range(count, len(original))),
                batch_size=64, shuffle=False)
        item["aux"], item["aux_meta"] = prepare_aux(
            original.y[:count], float(item["scaler_y"].mean_[0]),
            float(item["scaler_y"].scale_[0]), window_starts=item["train_starts"][:count])
        item["phase_manifest"] = {
            "household": item["household"], "phase": phase,
            "original_train_windows": len(original), "fit_windows": count,
            "internal_validation_windows": len(original) - count,
            "fit_x_sha256": array_hash(original.x[:count].numpy()),
            "fit_y_sha256": array_hash(original.y[:count].numpy()),
            "aux_source": "fit_windows_only", "author_manifest": item["manifest"],
            "preprocessing": item["manifest"]["preprocessing"],
        }
    return data


def dataframe_to_client(frame, appliance, phase, household="synthetic", source=None,
                        max_train_windows=None, max_test_windows=None):
    """Reconstruct the original numeric protocol for one household.

    Scalers see only the phase's raw training prefix. They intentionally use
    visible training rows even when those rows do not belong to a valid window.
    Auxiliary loss statistics, in contrast, use only the selected fit windows.
    All returned tensors remain on CPU; loading data performs no training.
    """
    if phase not in ("development", "formal") or appliance == "Aggregate":
        raise ValueError("Invalid phase or target appliance")
    for cap in (max_train_windows, max_test_windows):
        if cap is not None and (type(cap) is not int or cap <= 0):
            raise ValueError("Smoke window caps must be positive integers")
    raw = frame[["Aggregate", appliance]].to_numpy(dtype=np.float64, copy=True)
    if np.isinf(raw).any():
        raise ValueError("Infinite source power values")
    outer, inner = split_boundaries(len(raw))
    fit_end = inner if phase == "development" else outer
    if fit_end <= 0 or not np.isfinite(raw[:fit_end]).any(axis=0).all():
        raise ValueError("No valid train/test windows or finite scaler fit rows for " + household)

    fit_raw = raw[:fit_end]
    scaler_x = StandardScaler().fit(fit_raw[:, :1])
    scaler_y = StandardScaler().fit(fit_raw[:, 1:])
    normalized = np.column_stack((scaler_x.transform(raw[:, :1])[:, 0],
                                  scaler_y.transform(raw[:, 1:])[:, 0]))
    original_train_starts = window_starts(raw, 0, outer)
    if phase == "development":
        fit_starts = original_train_starts[original_train_starts + WINDOW <= inner]
        validation_starts = original_train_starts[original_train_starts >= inner]
    else:
        fit_starts = original_train_starts
        validation_starts = np.empty(0, dtype=np.int64)
    test_starts = window_starts(raw, outer, len(raw))
    full_counts = {"fit": len(fit_starts), "validation": len(validation_starts),
                   "train": len(fit_starts) + len(validation_starts), "test": len(test_starts)}
    if max_train_windows is not None:
        fit_cap = int(0.9 * max_train_windows) if phase == "development" else max_train_windows
        validation_cap = max_train_windows - fit_cap
        fit_starts = fit_starts[:fit_cap]
        validation_starts = validation_starts[:validation_cap]
    if max_test_windows is not None:
        test_starts = test_starts[:max_test_windows]
    if not len(fit_starts) or not len(test_starts):
        raise ValueError("No valid train/test windows for " + household)
    if phase == "development" and not len(validation_starts):
        raise ValueError("Insufficient training windows for internal validation: " + household)
    train_starts = np.concatenate((fit_starts, validation_starts))
    train_x, train_y = _windows(normalized, train_starts)
    test_x, test_y = _windows(normalized, test_starts)
    if not all(np.isfinite(a).all() for a in (train_x, train_y, test_x, test_y)):
        raise ValueError("Nonfinite normalized windows for " + household)
    if np.any(np.diff(train_starts) < WINDOW):
        raise AssertionError("Training/validation windows overlap")
    preprocessing = {
        "protocol": PROTOCOL, "phase": phase,
        "fit_scope": "development_fit_raw_segment" if phase == "development" else "outer_training_raw_segment",
        "fit_raw_row_interval": [0, fit_end],
        "fit_raw_values_sha256": array_hash(fit_raw),
        "fit_finite_rows_x": int(np.isfinite(fit_raw[:, 0]).sum()),
        "fit_finite_rows_y": int(np.isfinite(fit_raw[:, 1]).sum()),
        "test_statistics_used": False, "validation_statistics_used": False,
        "missingness_policy": "no time compression; original outer-train starts filtered at fixed dev boundary; purge crossing windows",
        "boundary_policy": "raw count only: outer=int(.8*N); dev=int(.9*((outer-1)//576))*576",
        "outer_train_raw_row_interval": [0, outer],
        "outer_test_raw_row_interval": [outer, len(raw)],
        "development_validation_raw_row_interval": [inner, outer] if phase == "development" else None,
        "scaler_includes_visible_nonwindow_rows": True,
        "development_crossing_windows_purged": int(len(original_train_starts) - full_counts["train"]),
        "fit_window_support": _support_manifest(fit_starts, [0, fit_end]),
        "validation_window_support": _support_manifest(validation_starts, [inner, outer]) if phase == "development" else None,
        "test_window_support": _support_manifest(test_starts, [outer, len(raw)]),
    }
    manifest = {
        "protocol": PROTOCOL, "household": household, "appliance": appliance,
        "source": source, "rows": len(raw), "raw_values_sha256": array_hash(raw),
        "split_index": outer, "development_split_index": inner if phase == "development" else None,
        "train_fraction": 0.8, "window": WINDOW, "stride_after_dropna": WINDOW,
        "last_possible_window_omitted": True,
        "internal_fit_final_window_can_end_at_boundary": phase == "development",
        "cross_boundary_windows": False, "scaler_fit": "phase_training_only",
        "known_test_distribution_leakage": False, "preprocessing": preprocessing,
        "scaler_x": scaler_manifest(scaler_x), "scaler_y": scaler_manifest(scaler_y),
        "full_window_counts": full_counts, "train_windows": len(train_starts),
        "fit_windows": len(fit_starts), "validation_windows": len(validation_starts),
        "test_windows": len(test_starts), "train_starts_sha256": array_hash(train_starts),
        "test_starts_sha256": array_hash(test_starts),
        "tensor_hashes": {name: array_hash(value) for name, value in
                         (("train_x", train_x), ("train_y", train_y),
                          ("test_x", test_x), ("test_y", test_y))},
    }
    train_dataset = WindowDataset(train_x, train_y)
    test_dataset = WindowDataset(test_x, test_y)
    item = {
        "household": household, "train_dataset": train_dataset, "test_dataset": test_dataset,
        "scaler_x": scaler_x, "scaler_y": scaler_y, "train_starts": train_starts,
        "test_starts": test_starts,
        "train_loader": torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True),
        "test_loader": torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False),
        "manifest": manifest,
    }
    prepare_phase([item], phase)
    return item


def load_clients(data_root, dataset, appliance, phase="formal", scope_path=None,
                 max_train_windows=None, max_test_windows=None, max_clients=None,
                 verify=True):
    """Load households in the frozen scope order, without architecture search.

    data_root contains the REDD/ and REFIT/ household CSV directories.
    verify=True checks supplied files against the released hashes, row counts,
    and headers. Actual hashes are recorded even when verification is disabled.
    Any explicit cap marks the returned scope metadata as a smoke run.
    """
    if phase not in ("development", "formal"):
        raise ValueError("Invalid phase")
    if type(verify) is not bool:
        raise ValueError("verify must be a boolean")
    for cap in (max_clients, max_train_windows, max_test_windows):
        if cap is not None and (type(cap) is not int or cap <= 0):
            raise ValueError("Smoke data caps must be positive integers")
    scope_file = Path(scope_path) if scope_path is not None else DEFAULT_SCOPE
    scope = json.loads(scope_file.read_text(encoding="utf-8-sig"))
    if scope.get("schema") != "nilm_f_data_scope_v1" or scope.get("window") != WINDOW:
        raise ValueError("Unsupported data scope or window length")
    matches = [task for task in scope["tasks"]
               if task["dataset"] == dataset and appliance in
               (task.get("canonical"), task["source_appliance"])]
    if len(matches) != 1:
        raise ValueError("Unknown or ambiguous task: " + str(dataset) + "/" + str(appliance))
    task = matches[0]
    households = task["households"]
    if (not households or len(households) != len(set(households)) or
            task["household_count"] != len(households)):
        raise ValueError("Invalid frozen household membership")
    files = {entry["path"]: entry for entry in scope["files"]}
    if len(files) != len(scope["files"]):
        raise ValueError("Duplicate source file in scope")
    if max_clients is not None:
        households = households[:max_clients]
    root = Path(data_root).resolve()
    caps = {"max_clients": max_clients, "max_train_windows": max_train_windows,
            "max_test_windows": max_test_windows}
    smoke = any(cap is not None for cap in caps.values())
    clients = []
    for household in households:
        relative = Path(dataset) / (household + ".csv")
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Source files must stay within data_root")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            raise ValueError("Source files must stay within data_root")
        expected = files.get(relative.as_posix())
        if expected is None:
            raise ValueError("Household file is missing from the frozen scope: " + str(relative))
        before = path.stat()
        digest = file_sha256(path)
        if verify and (before.st_size != expected["bytes"] or digest != expected["sha256"]):
            raise ValueError("Source CSV does not match the frozen scope: " + str(relative))
        columns = pd.read_csv(path, nrows=0).columns.tolist()
        frame = pd.read_csv(path, usecols=["Aggregate", task["source_appliance"]])
        if verify and (columns != expected["columns"] or len(frame) != expected["rows"]):
            raise ValueError("Source CSV rows or columns differ from the frozen scope: " + str(relative))
        source = {"path": str(path), "bytes": before.st_size,
                  "mtime_ns": before.st_mtime_ns, "sha256": digest}
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError("Source CSV changed while being read")
        item = dataframe_to_client(
            frame, task["source_appliance"], phase, household, source,
            max_train_windows=max_train_windows, max_test_windows=max_test_windows)
        item["scope_manifest"] = {
            "scope_sha256": file_sha256(scope_file), "task_id": task["task_id"],
            "dataset": dataset, "source_appliance": task["source_appliance"],
            "canonical": task["canonical"], "phase": phase,
            "household": household, "smoke": smoke, "data_caps": dict(caps),
            "frozen_source_verified": verify,
        }
        clients.append(item)
    return clients
