"""Watt-domain adaptive Huber loss with bounded local thresholds."""

import hashlib
import math

import numpy as np
import torch
import torch.nn.functional as F


ON_THRESHOLD_W = 50.0
ON_QUANTILE_SAMPLE_LIMIT = 65536
ROBUST_EMA = 0.95
ROBUST_UPPER_FACTOR = 10.0
ROBUST_SAMPLE_LIMIT = 2048


def _deterministic_on_quantile(array, mean, scale, n_on):
    if n_on == 0:
        return None, 0
    count = min(n_on, ON_QUANTILE_SAMPLE_LIMIT)
    ranks = np.linspace(0, n_on - 1, count, dtype=np.int64)
    values = np.empty(count, dtype=np.float64)
    seen, filled = 0, 0
    for row in array:
        power = row.astype(np.float64) * scale + mean
        active = power[power > ON_THRESHOLD_W]
        end = int(np.searchsorted(ranks, seen + len(active), side="left"))
        if end > filled:
            values[filled:end] = active[ranks[filled:end] - seen]
            filled = end
        seen += len(active)
    if filled != count or seen != n_on:
        raise RuntimeError("Training labels changed while preparing statistics")
    return float(np.quantile(values, 0.75)), count


def prepare_aux(y_norm, target_mean, target_scale, *, window_starts=None):
    """Return ``(None, metadata)`` from existing CPU training windows only.

    F does not consume event arrays, so no auxiliary tensors are retained.
    Historical ``author_mean_w`` / ``author_scale_w`` metadata names identify
    the supplied training-only target scaler; they do not load an author model
    or fit any statistic to validation/test data. Invalid window offsets are
    rejected even though the loss needs no cross-window event representation.
    """
    if not torch.is_tensor(y_norm) or y_norm.device.type != "cpu":
        raise ValueError("prepare_aux requires existing CPU training labels")
    if y_norm.ndim != 2 or min(y_norm.shape) <= 0:
        raise ValueError("Expected nonempty [windows, time] labels")
    if not y_norm.is_floating_point():
        raise ValueError("Training labels must be floating point")
    mean, scale = float(target_mean), float(target_scale)
    if not math.isfinite(mean) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid training-only target scaler")
    array = y_norm.detach().numpy()
    n, width = array.shape
    if window_starts is None:
        starts, contiguous_links = None, 0
    else:
        if torch.is_tensor(window_starts):
            window_starts = window_starts.detach().cpu().numpy()
        starts = np.asarray(window_starts)
        if starts.shape != (n,) or starts.dtype.kind not in "iu":
            raise ValueError("window_starts must be an integer vector of existing window offsets")
        starts = starts.astype(np.int64, copy=False)
        if starts[0] < 0 or np.any(np.diff(starts) < width):
            raise ValueError("Windows must be chronological and non-overlapping")
        contiguous_links = int((np.diff(starts) == width).sum())
    n_on, power_sum, negative_points = 0, 0.0, 0
    # Preserve the frozen row-wise float64 reconstruction and summation order.
    for row in array:
        power = row.astype(np.float64) * scale + mean
        if not np.isfinite(power).all():
            raise ValueError("Nonfinite training labels")
        n_on += int((power > ON_THRESHOLD_W).sum())
        power_sum += float(power.sum())
        negative_points += int((power < 0).sum())
    q75, sampled_on = _deterministic_on_quantile(array, mean, scale, n_on)
    power_scale = max(1.0, q75 if q75 is not None else ON_THRESHOLD_W)
    total = n * width
    meta = {
        "schema": "nilm_f_adaptive_huber_v1",
        "label_source": "existing_training_windows_only",
        "n_windows": n, "window": width, "n_points": total,
        "n_on_points": n_on, "n_off_points": total - n_on,
        "n_contiguous_window_links": contiguous_links,
        "window_starts_sha256": hashlib.sha256(starts.tobytes()).hexdigest()
        if starts is not None else None,
        "on_threshold_w": ON_THRESHOLD_W, "on_power_q75_w": q75,
        "on_quantile_sample_count": sampled_on,
        "quantile_method": "linear_q75_of_evenly_spaced_training_on_ranks",
        "power_scale_w": power_scale, "training_mean_power_w": power_sum / total,
        "negative_label_points": negative_points,
        "author_mean_w": mean, "author_scale_w": scale,
        "huber_delta_w": max(1.0, 0.1 * power_scale),
        "huber_divisor": scale * scale, "huber_quadratic_factor": 0.5,
        "huber_name": "watt_huber_divided_by_original_author_scale_squared",
        "extra_model_parameters": 0, "extra_forward_or_backward_passes": 0,
        "new_training_statistic_test_access": False,
        "persistent_aux_array_bytes": 0,
    }
    return None, meta


def _robust_bounds(meta):
    lower = float(meta["huber_delta_w"])
    upper = lower * ROBUST_UPPER_FACTOR
    if not math.isfinite(lower) or lower <= 0 or not math.isfinite(upper):
        raise ValueError("Invalid training Huber scale")
    return lower, upper


def new_robust_state(meta):
    """Initialize one JSON-safe household state at the fixed upper bound."""
    _, upper = _robust_bounds(meta)
    return {"delta_w": upper, "updates": 0}


def _validate_robust_state(state, meta):
    lower, upper = _robust_bounds(meta)
    if not isinstance(state, dict) or set(state) != {"delta_w", "updates"}:
        raise ValueError("An explicit local robust state is required")
    delta = float(state["delta_w"])
    updates = state["updates"]
    if not math.isfinite(delta) or not lower <= delta <= upper:
        raise ValueError("Robust delta outside frozen training bounds")
    if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
        raise ValueError("Invalid robust update count")
    return delta, lower, upper


def _residual_quantile(abs_residual_w):
    """Detached q75 of at most 2048 evenly spaced pre-update residuals."""
    flat = abs_residual_w.detach().reshape(-1)
    count = min(flat.numel(), ROBUST_SAMPLE_LIMIT)
    if count == 0:
        raise ValueError("Empty residual population")
    if count < flat.numel():
        index = torch.arange(count, device=flat.device, dtype=torch.int64)
        index = torch.div(index * (flat.numel() - 1), count - 1, rounding_mode="floor")
        flat = flat.index_select(0, index)
    return torch.quantile(flat.float(), 0.75), count


def adaptive_huber_loss(pred, target, meta, state):
    """Return loss and detached statistics without changing local state.

    Call within the model's autocast context. Huber and quantile arithmetic are
    explicitly FP32, matching the frozen training implementation, including
    inverse-transform-before-subtraction and division by target scale squared.
    Predictions are not clipped in training. Update the state separately only
    after the optimizer step succeeds.
    """
    if pred.shape != target.shape or pred.ndim != 2 or not pred.numel():
        raise ValueError("Expected matching nonempty [batch, time] prediction and target")
    with torch.autocast(device_type=pred.device.type, enabled=False):
        prediction, target_float = pred.float(), target.float()
        mse = F.mse_loss(prediction, target_float)
        stats = {"mse": mse.detach(), "n_points": target.numel()}
        prediction_w = prediction * meta["author_scale_w"] + meta["author_mean_w"]
        target_w = target_float * meta["author_scale_w"] + meta["author_mean_w"]
        delta = _validate_robust_state(state, meta)[0]
        loss = F.huber_loss(prediction_w, target_w, delta=delta) / meta["huber_divisor"]
        residual_w = (prediction_w.detach() - target_w.detach()).abs()
        quantile, count = _residual_quantile(residual_w)
        stats.update(loss_point=loss.detach(), loss_sequence=0.0, delta_used_w=delta,
                     linear_fraction=(residual_w > delta).float().mean(),
                     residual_q75_w=quantile, residual_sample_count=count,
                     loss=loss.detach())
    return loss, stats


def update_robust_state(state, stats, meta, step_succeeded=True):
    """Apply the bounded EMA after a successful step; AMP skips are a no-op."""
    if not step_succeeded:
        return state
    delta, lower, upper = _validate_robust_state(state, meta)
    residual_q75 = float(stats["residual_q75_w"])
    if not math.isfinite(residual_q75) or residual_q75 < 0:
        raise FloatingPointError("Invalid training residual quantile")
    updated = ROBUST_EMA * delta + (1.0 - ROBUST_EMA) * residual_q75
    state.update(delta_w=min(upper, max(lower, updated)), updates=state["updates"] + 1)
    return state
