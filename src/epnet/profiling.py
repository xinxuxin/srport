from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .modules import WindowAttention


@dataclass(frozen=True)
class ModelProfile:
    parameters: int
    macs: int
    flops: int
    latency_ms: float


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _conv_macs(module: nn.Conv2d, output: Tensor) -> int:
    batch, out_channels, out_h, out_w = output.shape
    kernel_ops = (
        module.kernel_size[0]
        * module.kernel_size[1]
        * (module.in_channels // module.groups)
    )
    return batch * out_channels * out_h * out_w * kernel_ops


def _linear_macs(module: nn.Linear, inputs: Tensor) -> int:
    rows = int(inputs.numel() / inputs.shape[-1])
    return rows * module.in_features * module.out_features


def _layer_norm_macs(output: Tensor) -> int:
    return output.numel() * 5


def _attention_macs(module: WindowAttention, inputs: Tensor) -> int:
    batch_windows, tokens, channels = inputs.shape
    return 2 * batch_windows * tokens * tokens * channels


def profile_model(
    model: nn.Module,
    input_tensor: Tensor,
    warmup: int = 1,
    iters: int = 5,
) -> ModelProfile:
    macs: dict[str, int] = {"value": 0}
    hooks: list[torch.utils.hooks.RemovableHandle] = []

    def conv_hook(module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
        if isinstance(module, nn.Conv2d):
            macs["value"] += _conv_macs(module, output)

    def linear_hook(module: nn.Module, inputs: tuple[Tensor, ...], _output: Tensor) -> None:
        if isinstance(module, nn.Linear):
            macs["value"] += _linear_macs(module, inputs[0])

    def layer_norm_hook(_module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
        macs["value"] += _layer_norm_macs(output)

    def attention_hook(module: nn.Module, inputs: tuple[Tensor, ...], _output: Tensor) -> None:
        if isinstance(module, WindowAttention):
            macs["value"] += _attention_macs(module, inputs[0])

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_hook))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(linear_hook))
        elif isinstance(module, nn.LayerNorm):
            hooks.append(module.register_forward_hook(layer_norm_hook))
        elif isinstance(module, WindowAttention):
            hooks.append(module.register_forward_hook(attention_hook))

    with torch.no_grad():
        for _ in range(warmup):
            model(input_tensor)

        start = time.perf_counter()
        for _ in range(iters):
            model(input_tensor)
        latency_ms = (time.perf_counter() - start) * 1000.0 / iters

    for hook in hooks:
        hook.remove()

    parameters = count_parameters(model)
    return ModelProfile(
        parameters=parameters,
        macs=macs["value"],
        flops=macs["value"] * 2,
        latency_ms=latency_ms,
    )
