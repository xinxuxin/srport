from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .device import build_device_context, model_memory_format
from .model import load_checkpoint, load_model_from_checkpoint
from .profiling import profile_model
from .utils import load_image, pil_to_tensor, save_image, tensor_to_pil


def run_inference(
    checkpoint_path: Path,
    input_path: Path,
    output_path: Path,
    *,
    device: str = "auto",
) -> dict[str, Any]:
    device_context = build_device_context(device, amp_mode="off", channels_last=True)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device_context.device)
    model = load_model_from_checkpoint(checkpoint).to(device_context.device)
    ema_state = checkpoint.get("ema_state")
    if isinstance(ema_state, dict):
        model.load_state_dict(ema_state)
    if device_context.channels_last:
        model = model.to(  # type: ignore[call-overload]
            memory_format=model_memory_format(device_context)
        )
    model.eval()

    image = load_image(input_path)
    tensor = pil_to_tensor(image).unsqueeze(0).to(device_context.device)
    if device_context.channels_last:
        tensor = tensor.contiguous(memory_format=model_memory_format(device_context))

    with torch.inference_mode():
        prediction = model(tensor)[0]

    output_image = tensor_to_pil(prediction)
    save_image(output_image, output_path)
    profile = profile_model(model, tensor)

    result = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "input_resolution": [image.width, image.height],
        "output_resolution": [output_image.width, output_image.height],
        "upscale": model.config.upscale,
        "parameters": profile.parameters,
        "estimated_macs": profile.macs,
        "estimated_flops": profile.flops,
        "latency_ms": profile.latency_ms,
        "device": device_context.device.type,
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run EPNet inference on a single image.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(
        json.dumps(
            run_inference(
                args.checkpoint,
                args.input,
                args.output,
                device=args.device,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
