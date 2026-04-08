from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import torch
from torch.utils.data import DataLoader

from .data import EvaluationImageDataset
from .metrics import evaluate_prediction
from .model import EPNet, load_model_from_checkpoint


def evaluate(checkpoint_path: Path, hr_dir: Path) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = load_model_from_checkpoint(checkpoint)
    ema_state = checkpoint.get("ema_state")
    if isinstance(ema_state, dict):
        model.load_state_dict(ema_state)
    model.eval()

    scale = int(model.config.upscale)
    dataset = EvaluationImageDataset(hr_dir, scale)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    results = []
    with torch.no_grad():
        for name, lr, hr in loader:
            prediction = model(lr)
            metrics = evaluate_prediction(prediction[0], hr[0], shave=scale)
            results.append({"name": name[0], "psnr": metrics.psnr, "ssim": metrics.ssim})

    summary = {
        "checkpoint": str(checkpoint_path),
        "samples": len(results),
        "average_psnr": mean(item["psnr"] for item in results),
        "average_ssim": mean(item["ssim"] for item in results),
        "per_image": results,
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate EPNet on a high-resolution image folder.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hr-dir", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(json.dumps(evaluate(args.checkpoint, args.hr_dir), indent=2))


if __name__ == "__main__":
    main()
