"""An explicit analytical memory convention, separate from measured GPU peak."""
from .model import is_adapter_key, parameter_report


def estimate_training_memory(model, batch_size=64, adapter_only=False):
    """Parameters + Adam's two FP32 moments + logical forward FP16 activations.

    This is an accounting estimate, not allocated/reserved CUDA memory. It
    excludes gradients, convolution workspaces, allocator fragmentation,
    checkpoint boundary saves, and backward temporaries. Checkpointing changes
    the actual liveness of activations, so its memory benefit must be measured.
    Activation shapes enumerate each Conv/ReLU/FiLM output, pool, dense/
    ReLU/FiLM, and final output once. It is intentionally not a prediction of
    the autograd saved-tensor peak (which can share storage or recompute).
    """
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    params = parameter_report(model)
    updated = sum(p.numel() for name, p in model.named_parameters()
                  if not adapter_only or is_adapter_key(name))
    activations_per_sample = (5 * 3 * 32 * 576 + 32 * 72 + 3 * 128 + 576)
    parameter_bytes = params["total_bytes"]
    adam_bytes = updated * 4 * 2
    activation_bytes = batch_size * activations_per_sample * 2
    total = parameter_bytes + adam_bytes + activation_bytes
    return {
        "convention": "FP32 parameters + two FP32 Adam moments + logical FP16 forward outputs",
        "parameters_bytes": parameter_bytes, "adam_moments_bytes": adam_bytes,
        "logical_activation_bytes": activation_bytes,
        "total_bytes": total, "total_KiB": total / 1024.0,
        "batch_size": batch_size, "adapter_only": adapter_only,
        "is_measured_gpu_peak": False, "includes_gradients_or_cuda_workspace": False,
    }
