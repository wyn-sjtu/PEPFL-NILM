"""Shared-backbone FedAvg followed by private-FiLM fine-tuning.

One household model occupies the accelerator at a time. Client weights and
Adam states are cached on the host; checkpoints preserve complete rounds."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import uuid

import numpy as np
import torch

from .data import array_hash, canonical_hash, file_sha256, scaler_manifest
from .loss import adaptive_huber_loss, new_robust_state, update_robust_state
from .model import FilmCNN, is_adapter_key, parameter_report
from .memory import estimate_training_memory


SCHEMA = "nilm_f_training_v1"
DEFAULT_CONFIG = {
    "seed": 1, "rounds": 50, "finetune_passes": 30,
    "batch_size": 64, "lr": 0.0003, "device": "cuda",
    "checkpointing": True, "amp": True, "num_threads": 4,
    "phase": "formal", "dataset": "", "appliance": "", "smoke": False,
}


def normalize_config(config):
    if not isinstance(config, dict) or set(config) - set(DEFAULT_CONFIG):
        raise ValueError("Training config contains unsupported keys")
    result = {**DEFAULT_CONFIG, **config}
    for key in ("seed", "rounds", "finetune_passes", "batch_size", "num_threads"):
        if type(result[key]) is not int or result[key] < (0 if key == "seed" else 1):
            raise ValueError("Invalid integer training option: " + key)
    for key in ("checkpointing", "amp", "smoke"):
        if type(result[key]) is not bool:
            raise ValueError("Expected boolean training option: " + key)
    if not isinstance(result["lr"], (float, int)) or isinstance(result["lr"], bool) \
            or not math.isfinite(result["lr"]) or result["lr"] <= 0:
        raise ValueError("Learning rate must be finite and positive")
    result["lr"] = float(result["lr"])
    if result["phase"] not in ("development", "formal"):
        raise ValueError("Unknown training phase")
    if any(not isinstance(result[key], str) for key in ("dataset", "appliance", "device")):
        raise ValueError("Dataset, appliance and device must be strings")
    device = torch.device(result["device"])
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Only CPU smoke or CUDA training is supported")
    if device.type == "cpu" and not result["smoke"]:
        raise ValueError("CPU execution is reserved for smoke validation")
    if not result["smoke"] and (result["seed"], result["rounds"], result["finetune_passes"],
                                 result["batch_size"], result["lr"]) != (1, 50, 30, 64, 0.0003):
        raise ValueError("Formal reproduction uses the fixed 50+30, seed1, batch64, Adam3e-4 budget")
    return result


def tree_cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: tree_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [tree_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(tree_cpu(item) for item in value)
    return copy.deepcopy(value)


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in state.items():
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    try:
        with temporary.open("wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(path):
    """Load only a trusted locally generated checkpoint (pickle format)."""
    return torch.load(Path(path), map_location="cpu", weights_only=False)


class _RunLock:
    def __init__(self, path):
        self.stream = Path(path).open("a+b")

    def __enter__(self):
        try:
            if os.name == "nt":
                import msvcrt
                self.stream.seek(0)
                if not self.stream.read(1):
                    self.stream.write(b"0")
                    self.stream.flush()
                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError):
            self.stream.close()
            raise RuntimeError("Another trainer owns this output directory")
        return self

    def __exit__(self, *args):
        if os.name == "nt":
            import msvcrt
            self.stream.seek(0)
            msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        self.stream.close()


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _rng_state(device):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}


def _restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def make_loader(dataset, config, round_index, household_index):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config["seed"] + round_index * 10000 + household_index)
    return torch.utils.data.DataLoader(dataset, batch_size=config["batch_size"],
                                      shuffle=True, drop_last=False, num_workers=0,
                                      generator=generator)


def initialize(data, config):
    """Initialize one reusable model and independent CPU household states."""
    device = torch.device(config["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA training requested but no CUDA device is available")
    if not data or len({item["household"] for item in data}) != len(data):
        raise ValueError("Expected nonempty unique households")
    if any(not len(item["fit_dataset"]) for item in data):
        raise ValueError("Each household needs at least one training window")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config["seed"])
        model = FilmCNN(checkpointing=config["checkpointing"])
    model = model.to(device)
    initial = tree_cpu(model.state_dict())
    states = [{"household": item["household"], "model": tree_cpu(initial),
               "optimizer": None, "scaler": None, "robust": new_robust_state(item["aux_meta"]),
               "updates": 0, "attempted_steps": 0, "amp_skipped_steps": 0,
               "completed_rounds": 0} for item in data]
    return model, states, {"initial_model_sha256": state_hash(initial),
                           "parameter_report": parameter_report(model)}


def _set_trainability(model, adapter_only):
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not adapter_only or is_adapter_key(name))
        parameter.grad = None


def _optimizer(model, incoming, lr, adapter_only):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, foreach=False)
    if incoming:
        if adapter_only:
            ids = [pid for group in incoming["param_groups"] for pid in group["params"]]
            ordered = list(model.named_parameters())
            if len(ids) != len(ordered):
                raise ValueError("Saved Adam parameter mapping mismatch")
            allowed = {pid for pid, (name, _) in zip(ids, ordered) if is_adapter_key(name)}
            incoming = {"param_groups": copy.deepcopy(incoming["param_groups"]),
                        "state": {key: value for key, value in incoming["state"].items() if key in allowed}}
        optimizer.load_state_dict(incoming)
    if any(group["lr"] != lr for group in optimizer.param_groups):
        raise ValueError("Checkpoint changed the configured learning rate")
    return optimizer


def _fit_client(model, state, item, loader, config, round_index):
    device = torch.device(config["device"])
    model.load_state_dict(state["model"])
    adapter_only = round_index > config["rounds"]
    _set_trainability(model, adapter_only)
    model.train()
    optimizer = _optimizer(model, state.get("optimizer"), config["lr"], adapter_only)
    amp_enabled = config["amp"] and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    if state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    attempts = skipped = samples = points = 0
    loss_sum = 0.0
    order_hash = hashlib.sha256()
    _synchronize(device)
    started = time.perf_counter()
    for x, y, indices in loader:
        order_hash.update(indices.detach().cpu().to(torch.int64).contiguous().numpy().tobytes())
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=torch.float16):
            prediction = model(x)
            loss, stats = adaptive_huber_loss(prediction, y, item["aux_meta"], state["robust"])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite training loss: " + state["household"])
        before_scale = scaler.get_scale()
        scaler.scale(loss).backward()
        if not amp_enabled and any(parameter.grad is not None and
                not bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()):
            raise FloatingPointError("Nonfinite FP32 gradients")
        scaler.step(optimizer)
        scaler.update()
        step_skipped = scaler.get_scale() < before_scale
        update_robust_state(state["robust"], stats, item["aux_meta"], step_succeeded=not step_skipped)
        attempts += 1
        skipped += int(step_skipped)
        samples += y.shape[0]
        points += y.numel()
        loss_sum += float(loss.detach()) * y.numel()
    _synchronize(device)
    elapsed = time.perf_counter() - started
    if attempts == 0:
        raise ValueError("No training batches")
    state["model"] = tree_cpu(model.state_dict())
    state["optimizer"] = tree_cpu(optimizer.state_dict())
    state["scaler"] = copy.deepcopy(scaler.state_dict())
    if not all(bool(torch.isfinite(value).all()) for value in state["model"].values()):
        raise FloatingPointError("Nonfinite trained household parameters")
    state["updates"] += attempts - skipped
    state["attempted_steps"] += attempts
    state["amp_skipped_steps"] += skipped
    optimizer.zero_grad(set_to_none=True)
    return {"household": state["household"], "scope": "adapter" if adapter_only else "all",
            "loss": loss_sum / points, "n_samples": samples, "n_points": points,
            "attempted_steps": attempts, "optimizer_updates": attempts - skipped,
            "amp_skipped_steps": skipped, "sampler_order_sha256": order_hash.hexdigest(),
            "optimizer_loop_seconds": elapsed, "robust": copy.deepcopy(state["robust"])}


def aggregate_clients(states):
    """Unweighted household mean on CPU; private FiLM and states stay local."""
    if not states:
        raise ValueError("No clients to aggregate")
    keys = [name for name in states[0]["model"] if not is_adapter_key(name)]
    averaged = {}
    for name in keys:
        values = [state["model"][name] for state in states]
        if any(value.device.type != "cpu" for value in values):
            raise ValueError("Persistent client state must be on CPU")
        total = values[0].clone()
        if total.is_floating_point():
            for value in values[1:]:
                total.add_(value)
            total.div_(len(values))
        elif any(not torch.equal(total, value) for value in values[1:]):
            raise ValueError("Cannot average differing non-floating buffers")
        averaged[name] = total
    for state in states:
        for name, value in averaged.items():
            state["model"][name] = value.clone()
    payload = sum(value.numel() * value.element_size() for value in averaged.values())
    return {"payload_bytes_per_client": payload, "upload_bytes": payload * len(states),
            "download_bytes": payload * len(states), "aggregated_keys": len(keys)}


def train_round(model, states, data, config, round_index):
    """Execute one paired local pass and, in rounds 1..50, FedAvg broadcast."""
    if not 1 <= round_index <= config["rounds"] + config["finetune_passes"]:
        raise ValueError("Invalid round index")
    if len(states) != len(data) or not states:
        raise ValueError("Client state and data differ")
    for state, item in zip(states, data):
        if state["household"] != item["household"] or state["completed_rounds"] != round_index - 1:
            raise ValueError("Household order or round does not match saved state")
    device = torch.device(config["device"])
    _synchronize(device)
    started = time.perf_counter()
    rows = [_fit_client(model, state, item, make_loader(item["fit_dataset"], config, round_index, index),
                        config, round_index) for index, (state, item) in enumerate(zip(states, data))]
    aggregation_started = time.perf_counter()
    aggregation = {"payload_bytes_per_client": 0, "upload_bytes": 0,
                   "download_bytes": 0, "aggregated_keys": 0}
    if round_index <= config["rounds"]:
        aggregation = aggregate_clients(states)
    aggregation_seconds = time.perf_counter() - aggregation_started
    for state in states:
        state["completed_rounds"] = round_index
    _synchronize(device)
    elapsed = time.perf_counter() - started
    sums = {key: sum(row[key] for row in rows) for key in
            ("n_samples", "n_points", "attempted_steps", "optimizer_updates", "amp_skipped_steps")}
    if sums["n_samples"] != sum(len(item["fit_dataset"]) for item in data):
        raise AssertionError("A local pass did not cover each training sample exactly once")
    return {"round": round_index, "stage": "federated" if round_index <= config["rounds"] else "finetune",
            "train_sync_wall_seconds": elapsed, "aggregation_seconds": aggregation_seconds,
            "household_training": rows, **sums, **aggregation}


def _portable(value):
    if isinstance(value, dict):
        return {key: _portable(item) for key, item in value.items() if key not in ("path", "mtime_ns", "project_root", "author_root")}
    if isinstance(value, list):
        return [_portable(item) for item in value]
    return value


def _data_identity(data, config=None):
    records, scalers = [], {}
    for item in data:
        household = item["household"]
        manifest = item["manifest"]
        provenance = manifest["preprocessing"]
        if (config is not None and provenance["phase"] != config["phase"]) or provenance["protocol"] != "train_only_v3" \
                or provenance["test_statistics_used"] is not False \
                or provenance["validation_statistics_used"] is not False:
            raise ValueError("Only matching phase-training-only preprocessing is permitted")
        if config is not None and item.get("scope_manifest", {}).get("smoke") and not config["smoke"]:
            raise ValueError("Capped data cannot be labelled a full reproduction")
        count = len(item["fit_dataset"])
        original = item["train_dataset"]
        if count != manifest["fit_windows"]:
            raise ValueError("Training window count differs from the data manifest")
        scalers[household] = {"x": scaler_manifest(item["scaler_x"]),
                              "y": scaler_manifest(item["scaler_y"])}
        if any(scalers[household][key] != manifest["scaler_" + key] for key in ("x", "y")):
            raise ValueError("Scaler changed after preprocessing")
        records.append({"household": household, "manifest": manifest,
                        "scope_manifest": item.get("scope_manifest"),
                        "fit_x_sha256": array_hash(original.x[:count].detach().cpu().numpy()),
                        "fit_y_sha256": array_hash(original.y[:count].detach().cpu().numpy()),
                        "fit_starts_sha256": array_hash(item["train_starts"][:count]),
                        "aux_meta": item["aux_meta"]})
    records = _portable(records)
    return canonical_hash(records), records, scalers


def data_fingerprint(clients):
    """Hash data values, scalers, source SHA and window support, not file locations."""
    return _data_identity(clients)[0]


def _implementation_fingerprint():
    directory = Path(__file__).resolve().parent
    hashes = {name: file_sha256(directory / name) for name in ("model.py", "loss.py", "data.py", "memory.py", "training.py")}
    return canonical_hash(hashes), hashes


def _hardware(device):
    record = {"device": str(device), "torch": str(torch.__version__),
              "cuda": str(torch.version.cuda), "cudnn": torch.backends.cudnn.version()}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        record.update(gpu_name=properties.name, gpu_total_memory_bytes=properties.total_memory,
                      gpu_uuid=str(getattr(properties, "uuid", "unavailable")))
    return record


def train(clients, config, output_dir, *, resume=False):
    """Train final F and return a JSON-safe summary including model_path.

    ``clients`` comes from data.load_clients. A completed run is returned only
    with resume=True and matching configuration/data/source fingerprints. The
    lightweight models.pt bundle is suitable for evaluation; latest.pt also
    contains Adam, AMP, threshold and RNG state for trusted local resumption.
    """
    config = normalize_config(config)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with _RunLock(output / ".training.lock"):
        return _train_locked(clients, config, output, resume)


def _train_locked(data, config, output, resume):
    began = time.perf_counter()
    device = torch.device(config["device"])
    torch.set_num_threads(config["num_threads"])
    if device.type == "cuda":
        torch.cuda.set_device(device)
    fingerprint, manifest, scalers = _data_identity(data, config)
    implementation_hash, sources = _implementation_fingerprint()
    latest, model_path = output / "latest.pt", output / "models.pt"
    identity = {"schema": SCHEMA, "config": config, "data_fingerprint": fingerprint,
                "implementation_sha256": implementation_hash}
    if any(path.exists() for path in (latest, model_path, output / "training_config.json", output / "training_result.json")) and not resume:
        raise FileExistsError("Training output already exists; use resume=True or another directory")
    if (output / "training_config.json").exists():
        if json.loads((output / "training_config.json").read_text(encoding="utf-8")) != identity:
            raise ValueError("Resume configuration, training data or implementation changed")
    else:
        if resume:
            raise FileNotFoundError("No training configuration to resume")
        atomic_json(output / "training_config.json", identity)
        atomic_json(output / "data_manifest.json", manifest)
    model, states, initial = initialize(data, config)
    completed, history, attempts, checkpoint_seconds = 0, [], [], 0.0
    if latest.exists():
        saved = load_checkpoint(latest)
        if any(saved.get(key) != value for key, value in identity.items()) or saved["scalers"] != scalers:
            raise ValueError("Checkpoint provenance differs from current inputs")
        if saved["initial_model_sha256"] != initial["initial_model_sha256"]:
            raise ValueError("Initial model changed")
        states, completed, history = saved["clients"], saved["completed_rounds"], saved["history"]
        attempts = saved["attempts"]
        checkpoint_seconds = saved.get("checkpoint_seconds_before_last_save", 0.0)
        if len(history) != completed or [row["round"] for row in history] != list(range(1, completed + 1)) \
                or len(states) != len(data):
            raise ValueError("Malformed round checkpoint")
        for state, item in zip(states, data):
            if state["household"] != item["household"] or state["completed_rounds"] != completed:
                raise ValueError("Checkpoint household order or round mismatch")
        if completed > 0:
            saved_hashes = history[-1].get("client_model_hashes")
            actual_hashes = {state["household"]: state_hash(state["model"]) for state in states}
            if saved_hashes != actual_hashes:
                raise ValueError("Checkpoint model weights differ from the last complete round")
        _restore_rng(saved["rng"])
    total_rounds = config["rounds"] + config["finetune_passes"]
    if not 0 <= completed <= total_rounds:
        raise ValueError("Checkpoint exceeds the training budget")
    result_path = output / "training_result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if completed != total_rounds or result.get("identity") != identity \
                or file_sha256(model_path) != result.get("model_sha256") \
                or file_sha256(output / "history.json") != result.get("history_sha256"):
            raise ValueError("Completed training artifacts changed")
        return result
    if (output / "run_attempts.json").exists():
        durable_attempts = json.loads((output / "run_attempts.json").read_text(encoding="utf-8"))
        if not isinstance(durable_attempts, list) or durable_attempts[:len(attempts)] != attempts:
            raise ValueError("Run-attempt history differs from checkpoint")
        attempts = durable_attempts
    attempt = {"started_unix_seconds": time.time(), "resumed_from_round": completed,
               "hardware": _hardware(device)}
    attempts = list(attempts) + [attempt]
    atomic_json(output / "run_attempts.json", attempts)
    for round_index in range(completed + 1, total_rounds + 1):
        atomic_json(output / "status.json", {"status": "training", "round": round_index,
                                             "total_rounds": total_rounds, "phase": config["phase"]})
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        stats = train_round(model, states, data, config, round_index)
        stats["training_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
        stats["training_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None
        stats["client_model_hashes"] = {state["household"]: state_hash(state["model"]) for state in states}
        history.append(stats)
        payload = {**identity, "source_sha256": sources, "clients": states, "scalers": scalers,
                   "completed_rounds": round_index, "completed_passes": round_index, "history": history, "attempts": attempts,
                   "initial_model_sha256": initial["initial_model_sha256"],
                   "checkpoint_seconds_before_last_save": checkpoint_seconds, "rng": _rng_state(device)}
        started = time.perf_counter()
        atomic_checkpoint(latest, payload)
        checkpoint_seconds += time.perf_counter() - started
        atomic_json(output / "history.json", history)
        print(json.dumps({"phase": config["phase"], "round": round_index, "total_rounds": total_rounds}), flush=True)
    atomic_json(output / "history.json", history)
    bundle = {**identity, "completed_passes": total_rounds, "initial_model_sha256": initial["initial_model_sha256"],
              "clients": [{"household": state["household"], "model": state["model"]} for state in states],
              "scalers": scalers}
    if model_path.exists():
        prior = load_checkpoint(model_path)
        if any(prior.get(key) != value for key, value in identity.items()) or [state_hash(x["model"]) for x in prior["clients"]] != [state_hash(x["model"]) for x in states]:
            raise ValueError("Existing deployment model differs from the completed checkpoint")
    else:
        atomic_checkpoint(model_path, bundle)
    phase_times = {phase: sum(row["train_sync_wall_seconds"] for row in history if row["stage"] == phase)
                   for phase in ("federated", "finetune")}
    memory = {phase: {key: max(row[key] for row in history if row["stage"] == phase)
                      if device.type == "cuda" else None
                      for key in ("training_peak_allocated_bytes", "training_peak_reserved_bytes")}
              for phase in ("federated", "finetune")}
    result = {"schema": SCHEMA, "status": "completed", "identity": identity, "config": config,
              "completed_rounds": total_rounds, "completed_passes": total_rounds, "households": len(states),
              "data_fingerprint": fingerprint, **initial,
              "checkpoint_path": str(latest), "model_path": str(model_path),
              "model_sha256": file_sha256(model_path), "history_path": str(output / "history.json"),
              "history_sha256": file_sha256(output / "history.json"),
              "search_seconds": 0.0, "phase_training_seconds": phase_times,
              "training_memory": memory,
              "memory_estimates":{phase:estimate_training_memory(model, config["batch_size"], phase=="finetune")
                                  for phase in ("federated","finetune")},
              "timing": {"training_seconds": sum(phase_times.values()),
                         "worker_wall_seconds_this_attempt": time.perf_counter() - began,
                         "checkpoint_seconds": checkpoint_seconds, "run_attempts": len(attempts),
                         "checkpoint_seconds_note": "On resume this excludes the last successful checkpoint write of each previous attempt; do not treat restarted checkpoint timing as a complete I/O total",
                         "timing_valid_without_restart": len(attempts) == 1,
                         "gpu_exclusivity_verified": False,
                         "note": "Synchronized training includes transfers and FedAvg; evaluation/calibration occur separately; no independent GPU-interference monitor is provided"},
              "hardware": attempt["hardware"], "attempts": attempts,
              "amp_requested":config["amp"], "amp_effective":config["amp"] and device.type=="cuda",
              "memory_definition": "PyTorch allocated peak per training phase; reserved is allocator cache, not live tensor memory; CPU client caches and CUDA context are excluded; CPU smoke reports no GPU memory"}
    atomic_json(result_path, result)
    atomic_json(output / "status.json", {"status": "training_completed", "completed_rounds": total_rounds})
    return result
