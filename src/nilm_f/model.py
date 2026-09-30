"""Fixed 1D CNN with household-private FiLM.

The shared state excludes FiLM parameters for backbone-only aggregation."""

from collections import OrderedDict
from typing import Dict, Mapping

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


WINDOW_LENGTH = 576
MIN_ADAPTER_BYTES = 1024
MAX_ADAPTER_BYTES = 5 * 1024


class ChannelFiLM(nn.Module):
    """Identity-initialized affine channel adaptation with FP32 parameters."""

    def __init__(self, channels: int):
        super().__init__()
        # Deliberately avoid random initialization: enabling adaptation must
        # not change the random stream used to initialize the shared network.
        self.gamma = nn.Parameter(torch.ones(channels, dtype=torch.float32))
        self.beta = nn.Parameter(torch.zeros(channels, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = (1, -1) + (1,) * (x.ndim - 2)
        # Under AMP, retain FP16 activations while keeping master adapter
        # parameters and their accumulated gradients in FP32.
        gamma = self.gamma.to(dtype=x.dtype).view(shape)
        beta = self.beta.to(dtype=x.dtype).view(shape)
        return x * gamma + beta


class ConvBlock(nn.Module):
    def __init__(self, inputs: int, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(inputs, channels, kernel_size=7, padding=3)
        self.relu = nn.ReLU(inplace=False)
        self.film = ChannelFiLM(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.film(self.relu(self.conv(x)))


class DenseBlock(nn.Module):
    def __init__(self, inputs: int, hidden: int):
        super().__init__()
        self.linear = nn.Linear(inputs, hidden)
        self.relu = nn.ReLU(inplace=False)
        self.film = ChannelFiLM(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.film(self.relu(self.linear(x)))


def is_adapter_key(name: str) -> bool:
    """Whether a model parameter/state key belongs to a FiLM adapter."""
    parts = name.split(".")
    return len(parts) >= 2 and parts[-2] == "film" and parts[-1] in {
        "gamma", "beta"
    }


def _state_partition(model: nn.Module, adapted: bool, cpu: bool):
    result = OrderedDict()
    for name, value in model.state_dict().items():
        if is_adapter_key(name) == adapted:
            value = value.detach()
            if cpu:
                value = value.cpu()
            result[name] = value.clone()
    return result


def shared_state_dict(model: nn.Module, cpu: bool = True):
    """Return independent shared tensors, on CPU by default for FedAvg."""
    return _state_partition(model, adapted=False, cpu=cpu)


def adapter_state_dict(model: nn.Module, cpu: bool = True):
    """Return independent household-private FiLM tensors."""
    return _state_partition(model, adapted=True, cpu=cpu)


def load_shared(model: nn.Module, state: Mapping[str, torch.Tensor]) -> None:
    """Load a complete shared backbone without modifying local adapters.

    Validate the complete set and every tensor before copying, so an invalid
    upload cannot partially update the model. Full model state is deliberately
    rejected here: aggregation of adapters must be explicitly requested by the
    caller via ``model.load_state_dict`` instead.
    """
    current = model.state_dict()
    expected = {name for name in current if not is_adapter_key(name)}
    received = set(state)
    if received != expected:
        raise ValueError(
            "Shared state keys differ: missing={}, unexpected={}".format(
                sorted(expected - received), sorted(received - expected)
            )
        )
    for name in expected:
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError("Shared state entry is not a tensor: {}".format(name))
        if value.shape != current[name].shape:
            raise ValueError("Shared state shape differs for {}".format(name))
    with torch.no_grad():
        for name in expected:
            current[name].copy_(state[name])


def parameter_report(model: nn.Module) -> Dict[str, object]:
    """Count actual parameter storage, explicitly separating local adapters."""
    shared_count = adapter_count = shared_bytes = adapter_bytes = 0
    trainable_count = 0
    for name, value in model.named_parameters():
        count = value.numel()
        storage = count * value.element_size()
        if is_adapter_key(name):
            adapter_count += count
            adapter_bytes += storage
        else:
            shared_count += count
            shared_bytes += storage
        if value.requires_grad:
            trainable_count += count
    return {
        "total_parameters": shared_count + adapter_count,
        "shared_parameters": shared_count,
        "adapter_parameters": adapter_count,
        "trainable_parameters": trainable_count,
        "total_bytes": shared_bytes + adapter_bytes,
        "shared_bytes": shared_bytes,
        "adapter_bytes": adapter_bytes,
        "total_KiB": (shared_bytes + adapter_bytes) / 1024.0,
        "shared_KiB": shared_bytes / 1024.0,
        "adapter_KiB": adapter_bytes / 1024.0,
        "adapter_within_budget": (
            adapter_count == 0 or MIN_ADAPTER_BYTES <= adapter_bytes <= MAX_ADAPTER_BYTES
        ),
        "storage_unit": "KiB = 1024 bytes; parameter tensors only",
    }


class FilmCNN(nn.Module):
    """Five conv7 blocks plus a dense sequence head, with no second model.

    Inputs and outputs are [batch, 576]. The model predicts unconstrained
    normalized power: applying a nonnegative
    clamp to normalized power would prevent physical zero-power predictions.

    Default FiLM storage is 576 FP32 parameters = 2304 bytes = 2.25 KiB.
    Dense pooling preserves all 576 input positions across 72 adjacent bins.
    Checkpointing recomputes groups of at most two convolution blocks during
    backward. It is bypassed for inference and when gradients are disabled.
    """

    def __init__(
        self,
        adapter: str = "film",
        channels: int = 32,
        hidden: int = 128,
        pool_length: int = 72,
        checkpointing: bool = True,
    ):
        super().__init__()
        if adapter != "film" or (channels, hidden, pool_length) != (32, 128, 72):
            raise ValueError("The final F architecture is fixed: film/32/128/72")
        if any(type(value) is not int for value in (channels, hidden, pool_length)):
            raise ValueError("Architecture dimensions must be integers")
        self.adapter = adapter
        self.channels = channels
        self.hidden = hidden
        self.pool_length = pool_length
        self.checkpointing = bool(checkpointing)
        self.features = nn.ModuleList(
            [ConvBlock(1 if index == 0 else channels, channels) for index in range(5)]
        )
        self.pool = nn.AdaptiveAvgPool1d(pool_length)
        self.dense = DenseBlock(channels * pool_length, hidden)
        self.output = nn.Linear(hidden, WINDOW_LENGTH)

    def _feature_range(self, x: torch.Tensor, start: int, end: int) -> torch.Tensor:
        for index in range(start, end):
            x = self.features[index](x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != WINDOW_LENGTH:
            raise ValueError("FilmCNN expects input shape [batch, 576]")
        x = x.unsqueeze(1)
        if self.checkpointing and self.training and torch.is_grad_enabled():
            for start in range(0, len(self.features), 2):
                x = checkpoint(
                    self._feature_range,
                    x,
                    start,
                    min(start + 2, len(self.features)),
                    use_reentrant=False,
                )
        else:
            x = self._feature_range(x, 0, len(self.features))
        x = self.pool(x).flatten(1)
        return self.output(self.dense(x))

    def shared_state_dict(self, cpu: bool = True):
        return shared_state_dict(self, cpu=cpu)

    def adapter_state_dict(self, cpu: bool = True):
        return adapter_state_dict(self, cpu=cpu)

    def load_shared(self, state: Mapping[str, torch.Tensor]) -> None:
        load_shared(self, state)

    def parameter_report(self) -> Dict[str, object]:
        return parameter_report(self)
