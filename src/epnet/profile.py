"""Model profiling entry point for presentation-ready deployment metrics.

This module complements evaluation by answering a different question: not "how
good is the model?", but "how large and how expensive is it to run?".

The output bundle is used both by local experimentation and by the deployment
API to surface parameters, MACs, FLOPs, checkpoint size, and a lightweight
latency estimate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .model import load_checkpoint, load_model_from_checkpoint
from .profiling import count_parameters, profile_model
from .report import render_profile_markdown, write_report_bundle
from .utils import build_device_context, model_memory_format


def profile_checkpoint(
    checkpoint_path: Path,
    *,
    device: str = "auto",
    onnx_export_path: Path | None = None,
    output_json: Path | None = None,
    output_markdown: Path | None = None,
) -> dict[str, Any]:
    """Profile a checkpoint and optionally test ONNX export as part of the run."""
    device_context = build_device_context(device, amp_mode="off", channels_last=True)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device_context.device)
    model = load_model_from_checkpoint(checkpoint).to(device_context.device)
    ema_state = checkpoint.get("ema_state")
    if isinstance(ema_state, dict):
        model.load_state_dict(ema_state)
    if device_context.channels_last:
        model = model.to(memory_format=model_memory_format(device_context))  # type: ignore[call-overload]

    input_shape = model.profile_input_shape()
    sample = torch.rand(input_shape, device=device_context.device)
    if device_context.channels_last:
        sample = sample.contiguous(memory_format=model_memory_format(device_context))
    profile = profile_model(model, sample)
    checkpoint_size = checkpoint_path.stat().st_size
    # ``model_state_size`` captures only serialized tensor storage. The
    # estimated runtime memory is intentionally coarse and should be presented
    # as an engineering approximation rather than a measured peak footprint.
    model_state_size = sum(
        tensor.element_size() * tensor.nelement()
        for tensor in model.state_dict().values()
        if isinstance(tensor, torch.Tensor)
    )
    payload = {
        "checkpoint": str(checkpoint_path),
        "device": device_context.device.type,
        "parameters": profile.parameters,
        "macs": profile.macs,
        "flops": profile.flops,
        "latency_ms": profile.latency_ms,
        "checkpoint_size_bytes": checkpoint_size,
        "model_state_size_bytes": model_state_size,
        "estimated_memory_bytes": model_state_size * 2,
        "onnx_export": "not-run",
    }
    if onnx_export_path is not None:
        from .export import export_onnx

        export_result = export_onnx(checkpoint_path, onnx_export_path, device="cpu")
        payload["onnx_export"] = export_result["status"]
        payload["onnx_export_error"] = export_result["error"]
        payload["onnx_export_path"] = export_result["output_path"]
    write_report_bundle(
        payload,
        output_json=output_json,
        output_markdown=output_markdown,
        renderer=render_profile_markdown,
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser used by ``epnet-profile``."""
    parser = argparse.ArgumentParser(description="Profile an EPNet checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-markdown", type=Path, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    return parser


def main() -> None:
    """Command-line entry point for model profiling."""
    args = build_parser().parse_args()
    print(
        json.dumps(
            profile_checkpoint(
                args.checkpoint,
                device=args.device,
                output_json=args.output_json,
                output_markdown=args.output_markdown,
            ),
            indent=2,
        )
    )


__all__ = ["count_parameters", "profile_checkpoint"]
