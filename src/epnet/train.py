from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from .config import ModelConfig, TrainConfig
from .data import SyntheticPatternDataset, TrainImageFolderDataset
from .ema import ExponentialMovingAverage
from .model import EPNet, load_checkpoint
from .utils import ensure_dir, set_seed


def _cycle(loader: Iterable[tuple[Tensor, Tensor]]) -> Iterator[tuple[Tensor, Tensor]]:
    while True:
        yield from loader


def save_checkpoint(
    path: Path,
    model: EPNet,
    optimizer: Adam,
    ema: ExponentialMovingAverage,
    step: int,
    model_config: ModelConfig,
    train_config: TrainConfig,
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
        },
        path,
    )


def train(
    train_dir: Path | None,
    output_path: Path,
    model_config: ModelConfig,
    train_config: TrainConfig,
    synthetic_count: int = 0,
    resume_path: Path | None = None,
) -> Path:
    if train_config.scale != model_config.upscale:
        raise ValueError(
            "TrainConfig.scale "
            f"({train_config.scale}) must match "
            f"ModelConfig.upscale ({model_config.upscale})."
        )
    set_seed(train_config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EPNet(model_config).to(device)
    optimizer = Adam(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2),
        weight_decay=train_config.weight_decay,
    )
    criterion: nn.Module = nn.L1Loss()
    ema = ExponentialMovingAverage(model, train_config.ema_decay)
    start_step = 0

    if resume_path is not None:
        checkpoint = load_checkpoint(resume_path, map_location=device)
        model_state = checkpoint.get("model_state")
        if not isinstance(model_state, dict):
            raise ValueError("Checkpoint is missing model_state for resume.")
        model.load_state_dict(model_state)
        optimizer_state = checkpoint.get("optimizer_state")
        if isinstance(optimizer_state, dict):
            optimizer.load_state_dict(optimizer_state)
        ema_state = checkpoint.get("ema_state")
        if isinstance(ema_state, dict):
            ema.shadow.load_state_dict(ema_state)
        step_value = checkpoint.get("step", 0)
        if not isinstance(step_value, (int, float, str)):
            raise ValueError("Checkpoint step must be numeric or string-coercible.")
        start_step = int(step_value)

    if train_dir is not None:
        dataset: Dataset[tuple[Tensor, Tensor]] = TrainImageFolderDataset(
            train_dir,
            train_config.patch_size,
            train_config.scale,
        )
    else:
        synthetic_size = train_config.patch_size
        dataset = SyntheticPatternDataset(
            count=max(128, synthetic_count or train_config.batch_size * 8),
            hr_size=synthetic_size,
            scale=train_config.scale,
            seed=train_config.seed,
        )

    loader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        drop_last=True,
    )
    batches = _cycle(loader)
    model.train()

    for step in range(start_step + 1, train_config.total_steps + 1):
        lr_batch, hr_batch = next(batches)
        lr_batch = lr_batch.to(device)
        hr_batch = hr_batch.to(device)

        optimizer.zero_grad(set_to_none=True)
        prediction = model(lr_batch)
        loss = criterion(prediction, hr_batch)
        loss.backward()
        optimizer.step()
        ema.update(model)

        if step % train_config.log_every == 0 or step == 1:
            print(json.dumps({"step": step, "loss": round(float(loss.item()), 6)}))

        if step % train_config.save_every == 0 or step == train_config.total_steps:
            save_checkpoint(output_path, model, optimizer, ema, step, model_config, train_config)

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
    parser.add_argument("--output", type=Path, required=True, help="Checkpoint output path.")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--patch-size", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--embed-dim", type=int, default=40)
    parser.add_argument("--num-pfem", type=int, default=4)
    parser.add_argument("--synthetic-count", type=int, default=0)
    parser.add_argument("--resume", type=Path, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    model_config = ModelConfig(upscale=args.scale, embed_dim=args.embed_dim, num_pfem=args.num_pfem)
    train_config = TrainConfig(
        scale=args.scale,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        total_steps=args.steps,
    )
    train(
        args.train_dir,
        args.output,
        model_config,
        train_config,
        synthetic_count=args.synthetic_count,
        resume_path=args.resume,
    )


if __name__ == "__main__":
    main()
