from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from torch.utils.data import DataLoader

from .config import DataConfig, load_data_config
from .data import EvaluationImageDataset, build_validation_datasets
from .model import load_checkpoint, load_model_from_checkpoint
from .report import render_eval_markdown, write_report_bundle
from .utils import build_device_context, evaluate_prediction, model_memory_format


def _evaluate_dataset(
    checkpoint_path: Path,
    dataset: EvaluationImageDataset,
    *,
    device: str,
) -> dict[str, Any]:
    device_context = build_device_context(device, amp_mode="off", channels_last=True)
    checkpoint = load_checkpoint(checkpoint_path, map_location=device_context.device)
    model = load_model_from_checkpoint(checkpoint).to(device_context.device)
    ema_state = checkpoint.get("ema_state")
    if isinstance(ema_state, dict):
        model.load_state_dict(ema_state)
    if device_context.channels_last:
        model = model.to(memory_format=model_memory_format(device_context))  # type: ignore[call-overload]
    model.eval()

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    results = []
    with torch.inference_mode():
        for name, lr, hr in loader:
            lr = lr.to(device_context.device)
            if device_context.channels_last:
                lr = lr.contiguous(memory_format=model_memory_format(device_context))
            prediction = model(lr)
            metrics = evaluate_prediction(prediction[0].float().cpu(), hr[0], shave=dataset.scale)
            results.append({"name": name[0], "psnr": metrics.psnr, "ssim": metrics.ssim})
    return {
        "device": device_context.device.type,
        "samples": len(results),
        "average_psnr": mean(item["psnr"] for item in results) if results else 0.0,
        "average_ssim": mean(item["ssim"] for item in results) if results else 0.0,
        "per_image": results,
    }


def evaluate(
    checkpoint_path: Path,
    hr_dir: Path,
    *,
    device: str = "auto",
) -> dict[str, Any]:
    dataset = EvaluationImageDataset(hr_dir, scale=_infer_scale(checkpoint_path))
    result = _evaluate_dataset(checkpoint_path, dataset, device=device)
    return {
        "checkpoint": str(checkpoint_path),
        "device": result["device"],
        "samples": result["samples"],
        "average_psnr": result["average_psnr"],
        "average_ssim": result["average_ssim"],
        "per_image": result["per_image"],
    }


def _infer_scale(checkpoint_path: Path) -> int:
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    raw_config = checkpoint.get("model_config", {})
    if isinstance(raw_config, dict):
        return int(raw_config.get("upscale", 4))
    return 4


def evaluate_run(
    checkpoint_path: Path,
    datasets: dict[str, EvaluationImageDataset],
    *,
    device: str = "auto",
    output_json: Path | None = None,
    output_markdown: Path | None = None,
) -> dict[str, Any]:
    dataset_results = {
        name: _evaluate_dataset(checkpoint_path, dataset, device=device)
        for name, dataset in datasets.items()
    }
    summary = {
        "checkpoint": str(checkpoint_path),
        "device": next(iter(dataset_results.values()))["device"] if dataset_results else device,
        "samples": sum(int(result["samples"]) for result in dataset_results.values()),
        "datasets": dataset_results,
        "summary": {
            "average_psnr": mean(result["average_psnr"] for result in dataset_results.values())
            if dataset_results
            else 0.0,
            "average_ssim": mean(result["average_ssim"] for result in dataset_results.values())
            if dataset_results
            else 0.0,
        },
    }
    write_report_bundle(
        summary,
        output_json=output_json,
        output_markdown=output_markdown,
        renderer=render_eval_markdown,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate EPNet on one or more datasets.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--hr-dir", type=Path, default=None)
    parser.add_argument("--data-config", type=Path, default=None)
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
    args = build_parser().parse_args()
    if args.data_config is not None:
        data_config: DataConfig = load_data_config(args.data_config)
        datasets = build_validation_datasets(data_config, scale=data_config.scale)
        summary = evaluate_run(
            args.checkpoint,
            datasets,
            device=args.device,
            output_json=args.output_json,
            output_markdown=args.output_markdown,
        )
        print(json.dumps(summary, indent=2))
        return

    if args.hr_dir is None:
        raise ValueError("Either --hr-dir or --data-config must be provided.")
    summary = evaluate(args.checkpoint, args.hr_dir, device=args.device)
    if args.output_json is not None:
        write_report_bundle(
            summary,
            output_json=args.output_json,
            output_markdown=args.output_markdown,
            renderer=lambda payload: json.dumps(payload, indent=2),
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
