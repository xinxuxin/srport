from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from epnet.config import ModelConfig, TrainConfig  # noqa: E402
from epnet.train import train  # noqa: E402


def test_training_resume_advances_checkpoint_step(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "resume.pt"
    train(
        train_dir=None,
        output_path=checkpoint_path,
        model_config=ModelConfig(upscale=2),
        train_config=TrainConfig(scale=2, batch_size=2, total_steps=2, save_every=1, log_every=1),
        synthetic_count=32,
    )
    first_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert first_checkpoint["step"] == 2
    assert "optimizer_state" in first_checkpoint

    train(
        train_dir=None,
        output_path=checkpoint_path,
        model_config=ModelConfig(upscale=2),
        train_config=TrainConfig(scale=2, batch_size=2, total_steps=3, save_every=1, log_every=1),
        synthetic_count=32,
        resume_path=checkpoint_path,
    )
    resumed_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert resumed_checkpoint["step"] == 3
