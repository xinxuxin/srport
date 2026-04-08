from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .config import ModelConfig, TrainConfig, model_config_from_variant
from .data import synthetic_pattern_image
from .device import build_device_context, model_memory_format
from .evaluate import evaluate
from .model import load_checkpoint, load_model_from_checkpoint
from .profiling import profile_model
from .train import train
from .utils import ensure_dir


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    group: str
    variant: str
    description: str
    overrides: dict[str, Any]


def build_default_experiments() -> list[ExperimentSpec]:
    return [
        ExperimentSpec(
            name="variant_tiny",
            group="variant",
            variant="tiny",
            description="Tiny preset for minimum parameter budget.",
            overrides={},
        ),
        ExperimentSpec(
            name="variant_paper",
            group="variant",
            variant="paper",
            description="Paper-grounded default preset.",
            overrides={},
        ),
        ExperimentSpec(
            name="variant_balanced",
            group="variant",
            variant="balanced",
            description="Slightly wider model for capacity/performance tradeoff.",
            overrides={},
        ),
        ExperimentSpec(
            name="ablation_pfem2",
            group="ablation",
            variant="paper",
            description="Reduce PFEM depth from n=4 to n=2.",
            overrides={"num_pfem": 2},
        ),
        ExperimentSpec(
            name="ablation_shared_pfem",
            group="ablation",
            variant="paper",
            description="Share PFEM block weights across repeated applications.",
            overrides={"share_pfem_weights": True},
        ),
        ExperimentSpec(
            name="ablation_espm2",
            group="ablation",
            variant="paper",
            description="Reduce ESPM pyramid depth from 3 to 2.",
            overrides={"espm_levels": 2},
        ),
    ]


def create_synthetic_eval_folder(
    output_dir: Path,
    *,
    image_count: int = 6,
    size: int = 96,
    seed: int = 1234,
) -> Path:
    ensure_dir(output_dir)
    for index in range(image_count):
        image = synthetic_pattern_image(size=size, seed=seed + index)
        image.save(output_dir / f"eval_{index:02d}.png")
    return output_dir


def _profile_checkpoint(checkpoint_path: Path, device: str) -> dict[str, float | int | str]:
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
    return {
        "device": device_context.device.type,
        "parameters": profile.parameters,
        "macs": profile.macs,
        "flops": profile.flops,
        "latency_ms": round(profile.latency_ms, 4),
    }


def _build_model_config(spec: ExperimentSpec, scale: int) -> ModelConfig:
    base = model_config_from_variant(spec.variant, upscale=scale)
    if not spec.overrides:
        return base
    return ModelConfig(**{**base.to_dict(), **spec.overrides})


def run_experiment(
    spec: ExperimentSpec,
    *,
    output_dir: Path,
    scale: int,
    steps: int,
    synthetic_count: int,
    device: str,
    val_dir: Path,
) -> dict[str, Any]:
    model_config = _build_model_config(spec, scale)
    train_config = TrainConfig(
        scale=scale,
        patch_size=48,
        batch_size=2,
        total_steps=steps,
        save_every=steps,
        log_every=1,
        device=device,
        amp="off",
        val_every=max(steps // 2, 1),
        max_validation_images=6,
        checkpoint_history=1,
    )

    checkpoint_path = output_dir / f"{spec.name}.pt"
    started = time.perf_counter()
    train(
        train_dir=None,
        output_path=checkpoint_path,
        model_config=model_config,
        train_config=train_config,
        synthetic_count=synthetic_count,
        val_dir=val_dir,
    )
    duration_s = time.perf_counter() - started

    inference_checkpoint = checkpoint_path.with_name(f"{checkpoint_path.stem}_inference.pt")
    evaluation = evaluate(inference_checkpoint, val_dir, device=device)
    profile = _profile_checkpoint(inference_checkpoint, device)
    training_checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")

    validation_metrics = training_checkpoint.get("validation_metrics", {})
    if not isinstance(validation_metrics, dict):
        validation_metrics = {}

    result = {
        "name": spec.name,
        "group": spec.group,
        "variant": spec.variant,
        "description": spec.description,
        "model_config": model_config.to_dict(),
        "train_config": train_config.to_dict(),
        "train_wall_time_s": round(duration_s, 4),
        "final_checkpoint": str(checkpoint_path),
        "inference_checkpoint": str(inference_checkpoint),
        "best_psnr": training_checkpoint.get("best_psnr"),
        "latest_loss": training_checkpoint.get("latest_loss"),
        "training_validation": validation_metrics,
        "evaluation": evaluation,
        "profile": profile,
    }
    return result


def _result_sort_key(result: dict[str, Any]) -> tuple[str, float]:
    evaluation = result.get("evaluation", {})
    if isinstance(evaluation, dict):
        psnr = evaluation.get("average_psnr", 0.0)
        if isinstance(psnr, (int, float)):
            return str(result.get("group", "")), -float(psnr)
    return str(result.get("group", "")), 0.0


def render_markdown_report(
    results: list[dict[str, Any]],
    *,
    scale: int,
    steps: int,
    synthetic_count: int,
    device: str,
    eval_dir: Path,
) -> str:
    sorted_results = sorted(results, key=_result_sort_key)
    lines = [
        "# EPNet Ablation Results",
        "",
        "## Experiment Setup",
        "",
        f"- Device: `{device}`",
        f"- Scale: `x{scale}`",
        f"- Steps per run: `{steps}`",
        f"- Synthetic training samples: `{synthetic_count}`",
        f"- Validation folder: `{eval_dir}`",
        (
            "- Validation data: deterministic synthetic images generated locally "
            "for reproducible smoke-style comparison"
        ),
        "- Important note: these are engineering comparison runs, not paper benchmark claims",
        "",
    ]

    for group in ("variant", "ablation"):
        group_results = [item for item in sorted_results if item["group"] == group]
        if not group_results:
            continue
        title = "Variant Comparison" if group == "variant" else "Ablation Comparison"
        lines.extend(
            [
                f"## {title}",
                "",
                (
                    "| Name | Variant | Params | MACs | Latency (ms) | Avg PSNR | "
                    "Avg SSIM | Best PSNR | Train Time (s) | Notes |"
                ),
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for result in group_results:
            evaluation = result["evaluation"]
            profile = result["profile"]
            lines.append(
                "| "
                f"{result['name']} | "
                f"{result['variant']} | "
                f"{profile['parameters']:,} | "
                f"{profile['macs']:,} | "
                f"{float(profile['latency_ms']):.2f} | "
                f"{float(evaluation['average_psnr']):.4f} | "
                f"{float(evaluation['average_ssim']):.5f} | "
                f"{float(result.get('best_psnr') or 0.0):.4f} | "
                f"{float(result['train_wall_time_s']):.2f} | "
                f"{result['description']} |"
            )
        lines.append("")

    best_psnr = max(results, key=lambda item: float(item["evaluation"]["average_psnr"]))
    fastest = min(results, key=lambda item: float(item["profile"]["latency_ms"]))
    smallest = min(results, key=lambda item: int(item["profile"]["parameters"]))
    lines.extend(
        [
            "## Key Takeaways",
            "",
            (
                f"- Best local PSNR in this run set: `{best_psnr['name']}` "
                f"with `{float(best_psnr['evaluation']['average_psnr']):.4f}` dB."
            ),
            (
                f"- Fastest profiled checkpoint: `{fastest['name']}` "
                f"at `{float(fastest['profile']['latency_ms']):.2f}` ms."
            ),
            (
                f"- Smallest model: `{smallest['name']}` "
                f"with `{int(smallest['profile']['parameters']):,}` parameters."
            ),
            (
                "- Interpretation warning: because the training and validation data are "
                "synthetic and very small, these results are only appropriate for "
                "relative smoke-style comparisons on this machine."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run_ablation_suite(
    *,
    output_dir: Path,
    report_path: Path,
    json_path: Path,
    scale: int = 2,
    steps: int = 8,
    synthetic_count: int = 128,
    device: str = "auto",
) -> dict[str, Any]:
    ensure_dir(output_dir)
    eval_dir = create_synthetic_eval_folder(output_dir / "synthetic_eval")
    experiments = build_default_experiments()
    results = [
        run_experiment(
            spec,
            output_dir=output_dir,
            scale=scale,
            steps=steps,
            synthetic_count=synthetic_count,
            device=device,
            val_dir=eval_dir,
        )
        for spec in experiments
    ]

    payload = {
        "device": device,
        "scale": scale,
        "steps": steps,
        "synthetic_count": synthetic_count,
        "evaluation_dir": str(eval_dir),
        "experiments": [asdict(spec) for spec in experiments],
        "results": results,
    }
    ensure_dir(report_path.parent)
    report_path.write_text(
        render_markdown_report(
            results,
            scale=scale,
            steps=steps,
            synthetic_count=synthetic_count,
            device=device,
            eval_dir=eval_dir,
        ),
        encoding="utf-8",
    )
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run EPNet variant comparison and ablation experiments."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-path", type=Path, required=True)
    parser.add_argument("--json-path", type=Path, required=True)
    parser.add_argument("--scale", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--synthetic-count", type=int, default=128)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = run_ablation_suite(
        output_dir=args.output_dir,
        report_path=args.report_path,
        json_path=args.json_path,
        scale=args.scale,
        steps=args.steps,
        synthetic_count=args.synthetic_count,
        device=args.device,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
