from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from statistics import mean

import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from .config import MODEL_VARIANTS, ModelConfig, TrainConfig, model_config_from_variant
from .data import EvaluationImageDataset, SyntheticPatternDataset, TrainImageFolderDataset
from .device import (
    DeviceContext,
    autocast_context,
    build_device_context,
    configure_backend,
    create_grad_scaler,
    model_memory_format,
    move_optimizer_state,
)
from .ema import ExponentialMovingAverage
from .metrics import evaluate_prediction
from .model import EPNet, load_checkpoint
from .utils import ensure_dir, set_seed


def _cycle(loader: Iterable[tuple[Tensor, Tensor]]) -> Iterator[tuple[Tensor, Tensor]]:
    while True:
        yield from loader


def _checkpoint_snapshot_path(output_path: Path, step: int) -> Path:
    return output_path.with_name(f"{output_path.stem}_step{step}{output_path.suffix}")


def _best_checkpoint_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_best{output_path.suffix}")


def _inference_checkpoint_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_inference{output_path.suffix}")


def save_checkpoint(
    path: Path,
    model: EPNet,
    optimizer: Adam,
    ema: ExponentialMovingAverage,
    step: int,
    model_config: ModelConfig,
    train_config: TrainConfig,
    *,
    scaler_state: dict[str, object] | None = None,
    best_psnr: float | None = None,
    latest_loss: float | None = None,
    device_type: str,
    validation_metrics: dict[str, float] | None = None,
) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "step": step,
            "model_config": model_config.to_dict(),
            "train_config": train_config.to_dict(),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "ema_state": ema.shadow.state_dict(),
            "scaler_state": scaler_state,
            "best_psnr": best_psnr,
            "latest_loss": latest_loss,
            "device_type": device_type,
            "validation_metrics": validation_metrics or {},
        },
        path,
    )


def export_inference_checkpoint(
    output_path: Path,
    model_config: ModelConfig,
    train_config: TrainConfig,
    ema: ExponentialMovingAverage,
    step: int,
    *,
    best_psnr: float | None,
) -> Path:
    inference_path = _inference_checkpoint_path(output_path)
    ensure_dir(inference_path.parent)
    torch.save(
        {
            "step": step,
            "model_config": model_config.to_dict(),
            "train_config": train_config.to_dict(),
            "model_state": ema.shadow.state_dict(),
            "ema_state": ema.shadow.state_dict(),
            "best_psnr": best_psnr,
            "checkpoint_kind": "inference",
        },
        inference_path,
    )
    return inference_path


def _make_loader(
    dataset: Dataset[tuple[Tensor, Tensor]],
    train_config: TrainConfig,
    device_context: DeviceContext,
    *,
    shuffle: bool,
) -> DataLoader[tuple[Tensor, Tensor]]:
    if train_config.num_workers > 0:
        return DataLoader(
            dataset,
            batch_size=train_config.batch_size,
            shuffle=shuffle,
            num_workers=train_config.num_workers,
            drop_last=shuffle,
            pin_memory=device_context.device.type == "cuda",
            persistent_workers=True,
            prefetch_factor=2,
        )
    return DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=shuffle,
        num_workers=train_config.num_workers,
        drop_last=shuffle,
        pin_memory=device_context.device.type == "cuda",
    )


def _prepare_batch(
    lr_batch: Tensor,
    hr_batch: Tensor,
    device_context: DeviceContext,
) -> tuple[Tensor, Tensor]:
    memory_format = model_memory_format(device_context)
    lr = lr_batch.to(
        device_context.device,
        non_blocking=device_context.non_blocking,
    ).contiguous(memory_format=memory_format)
    hr = hr_batch.to(
        device_context.device,
        non_blocking=device_context.non_blocking,
    ).contiguous(memory_format=memory_format)
    return lr, hr


def _retain_checkpoint_history(output_path: Path, keep: int) -> None:
    if keep <= 0:
        return
    history = sorted(output_path.parent.glob(f"{output_path.stem}_step*{output_path.suffix}"))
    for path in history[:-keep]:
        path.unlink(missing_ok=True)


def validate(
    model: nn.Module,
    hr_dir: Path,
    scale: int,
    device_context: DeviceContext,
    *,
    max_images: int = 8,
) -> dict[str, float]:
    dataset = EvaluationImageDataset(hr_dir, scale)
    results: list[dict[str, float]] = []
    model.eval()
    with torch.inference_mode():
        for index in range(min(len(dataset), max_images)):
            _, lr, hr = dataset[index]
            lr = lr.unsqueeze(0).to(device_context.device)
            if device_context.channels_last:
                lr = lr.contiguous(memory_format=torch.channels_last)
            with autocast_context(device_context):
                prediction = model(lr)[0]
            metrics = evaluate_prediction(prediction.float().cpu(), hr, shave=scale)
            results.append({"psnr": metrics.psnr, "ssim": metrics.ssim})
    model.train()
    return {
        "psnr": mean(item["psnr"] for item in results) if results else 0.0,
        "ssim": mean(item["ssim"] for item in results) if results else 0.0,
        "samples": float(len(results)),
    }


def _compile_if_requested(model: EPNet, train_config: TrainConfig) -> EPNet:
    if not train_config.compile_model or not hasattr(torch, "compile"):
        return model
    return torch.compile(model)  # type: ignore[return-value]


def train(
    train_dir: Path | None,
    output_path: Path,
    model_config: ModelConfig,
    train_config: TrainConfig,
    synthetic_count: int = 0,
    resume_path: Path | None = None,
    val_dir: Path | None = None,
) -> Path:
    if train_config.scale != model_config.upscale:
        raise ValueError(
            "TrainConfig.scale "
            f"({train_config.scale}) must match "
            f"ModelConfig.upscale ({model_config.upscale})."
        )
    set_seed(train_config.seed)
    device_context = build_device_context(
        preferred_device=train_config.device,
        amp_mode=train_config.amp,
        channels_last=train_config.channels_last,
    )
    configure_backend(device_context)

    model = EPNet(model_config).to(device_context.device)
    if device_context.channels_last:
        model = model.to(memory_format=torch.channels_last)  # type: ignore[call-overload]
    ema = ExponentialMovingAverage(model, train_config.ema_decay)
    training_model = _compile_if_requested(model, train_config)

    optimizer = Adam(
        training_model.parameters(),
        lr=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2),
        weight_decay=train_config.weight_decay,
    )
    criterion: nn.Module = nn.L1Loss()
    scaler = create_grad_scaler(device_context)
    start_step = 0
    best_psnr: float | None = None

    if resume_path is not None:
        checkpoint = load_checkpoint(resume_path, map_location=device_context.device)
        raw_model_config = checkpoint.get("model_config", {})
        if isinstance(raw_model_config, dict):
            checkpoint_scale = int(raw_model_config.get("upscale", model_config.upscale))
            if checkpoint_scale != model_config.upscale:
                raise ValueError(
                    "Resume checkpoint upscale does not match the current model configuration."
                )
        model_state = checkpoint.get("model_state")
        if not isinstance(model_state, dict):
            raise ValueError("Checkpoint is missing model_state for resume.")
        model.load_state_dict(model_state)
        optimizer_state = checkpoint.get("optimizer_state")
        if isinstance(optimizer_state, dict):
            optimizer.load_state_dict(optimizer_state)
            move_optimizer_state(optimizer, device_context.device)
        ema_state = checkpoint.get("ema_state")
        if isinstance(ema_state, dict):
            ema.shadow.load_state_dict(ema_state)
        scaler_state = checkpoint.get("scaler_state")
        if scaler is not None and isinstance(scaler_state, dict):
            scaler.load_state_dict(scaler_state)
        step_value = checkpoint.get("step", 0)
        if not isinstance(step_value, (int, float, str)):
            raise ValueError("Checkpoint step must be numeric or string-coercible.")
        start_step = int(step_value)
        checkpoint_best_psnr = checkpoint.get("best_psnr")
        if isinstance(checkpoint_best_psnr, (int, float)):
            best_psnr = float(checkpoint_best_psnr)

    if train_dir is not None:
        dataset: Dataset[tuple[Tensor, Tensor]] = TrainImageFolderDataset(
            train_dir,
            train_config.patch_size,
            train_config.scale,
        )
    else:
        dataset = SyntheticPatternDataset(
            count=max(128, synthetic_count or train_config.batch_size * 8),
            hr_size=train_config.patch_size,
            scale=train_config.scale,
            seed=train_config.seed,
        )

    loader = _make_loader(dataset, train_config, device_context, shuffle=True)
    batches = _cycle(loader)
    model.train()
    training_model.train()

    latest_loss = 0.0
    last_validation_metrics: dict[str, float] | None = None

    for step in range(start_step + 1, train_config.total_steps + 1):
        lr_batch, hr_batch = next(batches)
        lr_batch, hr_batch = _prepare_batch(lr_batch, hr_batch, device_context)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device_context):
            prediction = training_model(lr_batch)
            loss = criterion(prediction, hr_batch)

        if scaler is not None:
            scaler.scale(loss).backward()
            if train_config.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if train_config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip_norm)
            optimizer.step()
        ema.update(model)
        latest_loss = float(loss.item())

        if step % train_config.log_every == 0 or step == 1:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": round(latest_loss, 6),
                        "device": device_context.device.type,
                        "amp": device_context.amp_enabled,
                    }
                )
            )

        run_validation = (
            val_dir is not None
            and train_config.val_every > 0
            and (step % train_config.val_every == 0 or step == train_config.total_steps)
        )
        if run_validation:
            assert val_dir is not None
            last_validation_metrics = validate(
                ema.shadow,
                val_dir,
                train_config.scale,
                device_context,
                max_images=train_config.max_validation_images,
            )
            print(
                json.dumps(
                    {
                        "step": step,
                        "validation_psnr": round(last_validation_metrics["psnr"], 4),
                        "validation_ssim": round(last_validation_metrics["ssim"], 5),
                    }
                )
            )
            current_psnr = last_validation_metrics["psnr"]
            if train_config.save_best and (best_psnr is None or current_psnr >= best_psnr):
                best_psnr = current_psnr
                save_checkpoint(
                    _best_checkpoint_path(output_path),
                    model,
                    optimizer,
                    ema,
                    step,
                    model_config,
                    train_config,
                    scaler_state=scaler.state_dict() if scaler is not None else None,
                    best_psnr=best_psnr,
                    latest_loss=latest_loss,
                    device_type=device_context.device.type,
                    validation_metrics=last_validation_metrics,
                )

        if step % train_config.save_every == 0 or step == train_config.total_steps:
            save_checkpoint(
                output_path,
                model,
                optimizer,
                ema,
                step,
                model_config,
                train_config,
                scaler_state=scaler.state_dict() if scaler is not None else None,
                best_psnr=best_psnr,
                latest_loss=latest_loss,
                device_type=device_context.device.type,
                validation_metrics=last_validation_metrics,
            )
            snapshot_path = _checkpoint_snapshot_path(output_path, step)
            save_checkpoint(
                snapshot_path,
                model,
                optimizer,
                ema,
                step,
                model_config,
                train_config,
                scaler_state=scaler.state_dict() if scaler is not None else None,
                best_psnr=best_psnr,
                latest_loss=latest_loss,
                device_type=device_context.device.type,
                validation_metrics=last_validation_metrics,
            )
            _retain_checkpoint_history(output_path, train_config.checkpoint_history)

    inference_path = export_inference_checkpoint(
        output_path,
        model_config,
        train_config,
        ema,
        train_config.total_steps,
        best_psnr=best_psnr,
    )
    print(
        json.dumps(
            {
                "final_checkpoint": str(output_path),
                "inference_checkpoint": str(inference_path),
                "device": device_context.device.type,
                "best_psnr": best_psnr,
            }
        )
    )
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train EPNet for single-image super-resolution."
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=None,
        help="Directory of high-resolution images.",
    )
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=None,
        help="Optional validation HR directory.",
    )
    parser.add_argument("--output", type=Path, required=True, help="Checkpoint output path.")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--variant", type=str, default="paper", choices=sorted(MODEL_VARIANTS))
    parser.add_argument("--patch-size", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--num-pfem", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--synthetic-count", type=int, default=0)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
    )
    parser.add_argument("--amp", type=str, default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--val-every", type=int, default=0)
    parser.add_argument("--max-validation-images", type=int, default=8)
    parser.add_argument("--channels-last", action="store_true", default=True)
    parser.add_argument("--no-channels-last", action="store_false", dest="channels_last")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--checkpoint-history", type=int, default=2)
    parser.add_argument("--share-pfem-weights", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    model_config = model_config_from_variant(
        args.variant,
        upscale=args.scale,
        share_pfem_weights=args.share_pfem_weights,
    )
    if args.embed_dim is not None:
        model_config = ModelConfig(**{**model_config.to_dict(), "embed_dim": args.embed_dim})
    if args.num_pfem is not None:
        model_config = ModelConfig(**{**model_config.to_dict(), "num_pfem": args.num_pfem})
    if args.num_heads is not None:
        model_config = ModelConfig(**{**model_config.to_dict(), "num_heads": args.num_heads})

    train_config = TrainConfig(
        scale=args.scale,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        total_steps=args.steps,
        num_workers=args.num_workers,
        device=args.device,
        amp=args.amp,
        grad_clip_norm=args.grad_clip_norm,
        channels_last=args.channels_last,
        compile_model=args.compile_model,
        val_every=args.val_every,
        max_validation_images=args.max_validation_images,
        checkpoint_history=args.checkpoint_history,
    )
    train(
        args.train_dir,
        args.output,
        model_config,
        train_config,
        synthetic_count=args.synthetic_count,
        resume_path=args.resume,
        val_dir=args.val_dir,
    )


if __name__ == "__main__":
    main()
