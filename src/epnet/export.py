"""ONNX export helper for deployment-oriented EPNet artifacts.

This file does not define the deployment runtime itself. Instead, it provides a
repeatable way to test whether a trained checkpoint can be exported into an
interchange format that may later be consumed by edge runtimes such as
ONNXRuntime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .model import load_checkpoint, load_model_from_checkpoint
from .utils import build_device_context, ensure_dir, model_memory_format


def export_onnx(
    checkpoint_path: Path,
    output_path: Path,
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    """Export a checkpoint to ONNX and return a structured result payload.

    Implementation notes:
        - The function uses the repository's profile input shape as a stable
          export example tensor.
        - Dynamic height/width axes are requested, but actual downstream runtime
          compatibility still depends on the exported operator set.
    """
    device_context = build_device_context(device, amp_mode="off", channels_last=True)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device_context.device)
    model = load_model_from_checkpoint(checkpoint).to(device_context.device)
    ema_state = checkpoint.get("ema_state")
    if isinstance(ema_state, dict):
        model.load_state_dict(ema_state)
    if device_context.channels_last:
        model = model.to(memory_format=model_memory_format(device_context))  # type: ignore[call-overload]
    model.eval()

    dummy = torch.rand(model.profile_input_shape(), device=device_context.device)
    if device_context.channels_last:
        dummy = dummy.contiguous(memory_format=model_memory_format(device_context))

    ensure_dir(output_path.parent)
    try:
        # Export is wrapped in a broad ``try`` so the training and profiling
        # pipelines can record an actionable failure instead of crashing.
        torch.onnx.export(
            model,
            (dummy,),
            output_path,
            input_names=["lr"],
            output_names=["sr"],
            opset_version=17,
            dynamic_axes={"lr": {0: "batch", 2: "height", 3: "width"}, "sr": {0: "batch"}},
        )
        status = "success"
        error = None
    except Exception as exc:
        status = "failed"
        error = str(exc)
        output_path.unlink(missing_ok=True)
    return {
        "checkpoint": str(checkpoint_path),
        "output_path": str(output_path),
        "status": status,
        "error": error,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser used by ``epnet-export``."""
    parser = argparse.ArgumentParser(description="Export EPNet checkpoint to ONNX.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    return parser


def main() -> None:
    """Command-line entry point for ONNX export."""
    args = build_parser().parse_args()
    print(json.dumps(export_onnx(args.checkpoint, args.output, device=args.device), indent=2))


__all__ = ["export_onnx"]
