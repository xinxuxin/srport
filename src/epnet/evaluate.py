from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import torch
from torch.utils.data import DataLoader

from .data import EvaluationImageDataset
from .device import build_device_context, model_memory_format
from .metrics import evaluate_prediction
from .model import load_checkpoint, load_model_from_checkpoint


def evaluate(checkpoint_path: Path, hr_dir: Path, *, device: str = "auto") -> dict[str, object]:
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

    scale = int(model.config.upscale)
    dataset = EvaluationImageDataset(hr_dir, scale)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    results = []
    with torch.inference_mode():
        for name, lr, hr in loader:
            lr = lr.to(device_context.device)
            if device_context.channels_last:
                lr = lr.contiguous(memory_format=model_memory_format(device_context))
            prediction = model(lr)
            metrics = evaluate_prediction(prediction[0].float().cpu(), hr[0], shave=scale)
            results.append({"name": name[0], "psnr": metrics.psnr, "ssim": metrics.ssim})

    summary = {
        "checkpoint": str(checkpoint_path),
        "device": device_context.device.type,
        "samples": len(results),
        "average_psnr": mean(item["psnr"] for item in results),
        "average_ssim": mean(item["ssim"] for item in results),
        "per_image": results,
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate EPNet on a high-resolution image folder."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hr-dir", type=Path, required=True)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(evaluate(args.checkpoint, args.hr_dir, device=args.device), indent=2))


if __name__ == "__main__":
    main()
