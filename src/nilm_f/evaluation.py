"""Uncalibrated, household-macro evaluation in physical watts."""
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error

from .data import file_sha256, load_clients, scaler_manifest
from .model import FilmCNN


def metric(truth, prediction, household):
    """MAE matches the historical float32, multi-output sklearn reduction.

    F1 is calculated from all evaluated points within a household, with a
    strict >50 W threshold. It is then macro-averaged across households.
    """
    y, p = np.asarray(truth, dtype=np.float32), np.asarray(prediction, dtype=np.float32)
    if y.shape != p.shape or y.ndim != 2 or not y.size:
        raise ValueError("Expected equal, nonempty [windows, 576] arrays")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("Nonfinite predictions or labels")
    active_y, active_p = y > 50.0, p > 50.0
    tp = int(np.count_nonzero(active_y & active_p))
    fp = int(np.count_nonzero(~active_y & active_p))
    fn = int(np.count_nonzero(active_y & ~active_p))
    denominator = 2 * tp + fp + fn
    return {
        "household": household,
        "mae_w": float(mean_absolute_error(y, p)),
        "f1_percent": 100.0 * 2 * tp / denominator if denominator else 0.0,
        "tp": tp, "fp": fp, "fn": fn,
        "n_windows": int(len(y)), "n_points": int(y.size),
    }


def macro_average(rows):
    if not rows:
        raise ValueError("No households to evaluate")
    return {
        "mae_w": sum(row["mae_w"] for row in rows) / len(rows),
        "f1_percent": sum(row["f1_percent"] for row in rows) / len(rows),
        "n_households": len(rows), "n_points": sum(row["n_points"] for row in rows),
        "households": rows,
    }


def predict(model, state, item, split="test", device="cpu"):
    """Return prediction, target and aggregate arrays in watts, without AMP."""
    if split not in ("test", "validation"):
        raise ValueError("split must be test or validation")
    if split == "validation" and "validation_loader" not in item:
        raise ValueError("Formal data has no internal validation partition")
    dataset = item["test_dataset"] if split == "test" else item["validation_loader"].dataset
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=0,
        generator=torch.Generator().manual_seed(1))
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    predictions, targets, aggregates = [], [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch[:2]
            predictions.append(model(x.to(device)).float().cpu().numpy())
            targets.append(y.numpy())
            aggregates.append(x.numpy())
    p = item["scaler_y"].inverse_transform(np.concatenate(predictions))
    y = item["scaler_y"].inverse_transform(np.concatenate(targets))
    x = item["scaler_x"].inverse_transform(np.concatenate(aggregates))
    return np.maximum(p, 0).astype(np.float32), y, x


def evaluate(checkpoint, data_root, output_dir, device="cuda", save_predictions=False):
    """Evaluate a completed package checkpoint; never use test data for fitting."""
    checkpoint, output_dir = Path(checkpoint), Path(output_dir)
    # Only load trusted, locally produced checkpoints. PyTorch 2.0 stores Python
    # optimizer/config metadata with pickle in addition to tensor state.
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
    from .training import SCHEMA, data_fingerprint, normalize_config, state_hash
    if bundle.get("schema") != SCHEMA:
        raise ValueError("Expected a final-F package checkpoint")
    config = normalize_config(bundle["config"])
    phase = config["phase"]
    expected_passes = config["rounds"] + config["finetune_passes"]
    completed = bundle.get("completed_passes")
    if type(completed) is not int or completed != expected_passes:
        raise ValueError("Evaluation requires the completed, fixed-budget checkpoint")
    if "history" in bundle:
        history = bundle["history"]
        if len(history) != completed or any(
                history[-1]["client_model_hashes"].get(client["household"]) != state_hash(client["model"])
                for client in bundle["clients"]):
            raise ValueError("Checkpoint weights disagree with the final training record")
    caps = {"max_clients": 2, "max_train_windows": 12, "max_test_windows": 4} if config.get("smoke") else {}
    data = load_clients(data_root, config["dataset"], config["appliance"], phase=phase, **caps)
    states = bundle["clients"]
    if [c["household"] for c in states] != [c["household"] for c in data]:
        raise ValueError("Checkpoint and data household order differ")
    if bundle.get("data_fingerprint") != data_fingerprint(data):
        raise ValueError("Checkpoint preprocessing fingerprint differs from supplied data")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu for a CPU validation run")
    # Match the original training/evaluation worker even in a fresh process.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    model = FilmCNN(checkpointing=False).to(device)
    split = "validation" if phase == "development" else "test"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, start = [], time.perf_counter()
    for state, item in zip(states, data):
        household = item["household"]
        expected_scalers = {"x": scaler_manifest(item["scaler_x"]), "y": scaler_manifest(item["scaler_y"])}
        if bundle["scalers"][household] != expected_scalers:
            raise ValueError("Checkpoint scaler statistics differ: " + household)
        p, y, x = predict(model, state["model"], item, split=split, device=device)
        rows.append(metric(y, p, household))
        if save_predictions:
            starts = item["test_starts"] if split == "test" else item["train_starts"][item["manifest"]["fit_windows"]:]
            np.savez_compressed(output_dir / (household + ".npz"),
                                prediction_w=p, target_w=y, aggregate_w=x,
                                raw_window_start=starts)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    result = dict(macro_average(rows), schema="nilm_f_metrics_v1", method="F",
                  dataset=config["dataset"], appliance=config["appliance"],
                  phase=phase, split=split, seed=config["seed"], smoke=config.get("smoke", False),
                  aggregation="equal household macro-average", activation_rule="power > 50 W",
                  learned_calibration=False, inference_clipping="max(0, watts)",
                  evaluation_seconds=time.perf_counter() - start,
                  checkpoint_sha256=file_sha256(checkpoint), device=device)
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result
