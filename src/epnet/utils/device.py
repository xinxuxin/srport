"""Device and mixed-precision helpers shared across training and evaluation.

This file centralizes the repository's runtime policy for CPU, CUDA, and MPS so
the rest of the code can reason in terms of a single ``DeviceContext`` object.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DeviceContext:
    """Container describing how tensors and kernels should run on this machine."""

    device: torch.device
    amp_enabled: bool
    amp_dtype: torch.dtype | None
    channels_last: bool
    non_blocking: bool


def detect_best_device(preferred: str = "auto") -> torch.device:
    """Resolve ``auto`` / ``cpu`` / ``cuda`` / ``mps`` into a concrete device."""
    normalized = preferred.lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if normalized == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if normalized == "mps":
        if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            raise ValueError("MPS was requested but is not available.")
        return torch.device("mps")
    if normalized == "cpu":
        return torch.device("cpu")
    raise ValueError("Device must be one of: auto, cuda, mps, cpu.")


def build_device_context(
    preferred_device: str = "auto",
    amp_mode: str = "auto",
    channels_last: bool = True,
) -> DeviceContext:
    """Build the runtime policy used by train/eval/infer code.

    Important policy decisions:
        - AMP is enabled only on CUDA in this repository for stability.
        - channels-last is allowed on CUDA and MPS, where it may improve
          convolution throughput.
    """
    device = detect_best_device(preferred_device)
    normalized_amp = amp_mode.lower()
    if normalized_amp not in {"auto", "on", "off"}:
        raise ValueError("AMP mode must be one of: auto, on, off.")

    if normalized_amp == "off":
        amp_enabled = False
    elif normalized_amp == "on":
        if device.type != "cuda":
            raise ValueError("AMP is only enabled for CUDA in this training pipeline.")
        amp_enabled = True
    else:
        amp_enabled = device.type == "cuda"

    amp_dtype = torch.float16 if amp_enabled and device.type == "cuda" else None
    return DeviceContext(
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
        channels_last=channels_last and device.type in {"cuda", "mps"},
        non_blocking=device.type == "cuda",
    )


def autocast_context(
    context: DeviceContext,
) -> torch.amp.autocast_mode.autocast | nullcontext[None]:
    """Return the appropriate autocast context manager for the current device."""
    if not context.amp_enabled or context.amp_dtype is None:
        return nullcontext()
    return torch.autocast(device_type=context.device.type, dtype=context.amp_dtype)


def create_grad_scaler(context: DeviceContext) -> torch.cuda.amp.GradScaler | None:
    """Create a CUDA grad scaler when AMP is active."""
    if not context.amp_enabled or context.device.type != "cuda":
        return None
    return torch.cuda.amp.GradScaler()


def model_memory_format(context: DeviceContext) -> torch.memory_format:
    """Choose the preferred memory format for model and tensor storage."""
    if context.channels_last:
        return torch.channels_last
    return torch.contiguous_format


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    """Move optimizer tensors after loading a checkpoint onto the active device."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def configure_backend(context: DeviceContext) -> None:
    """Enable backend-specific speed knobs where they are safe."""
    if context.device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
