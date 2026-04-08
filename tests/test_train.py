from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from epnet.config import ModelConfig, TrainConfig, model_config_from_variant  # noqa: E402
from epnet.device import build_device_context  # noqa: E402
from epnet.train import train  # noqa: E402


def test_training_resume_advances_checkpoint_step(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "resume.pt"
    train(
        train_dir=None,
        output_path=checkpoint_path,
        model_config=ModelConfig(upscale=2),
        train_config=TrainConfig(
            scale=2,
            batch_size=2,
            total_steps=2,
            save_every=1,
            log_every=1,
            device="cpu",
        ),
        synthetic_count=32,
    )
    first_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert first_checkpoint["step"] == 2
    assert "optimizer_state" in first_checkpoint
    assert first_checkpoint["device_type"] == "cpu"

    train(
        train_dir=None,
        output_path=checkpoint_path,
        model_config=ModelConfig(upscale=2),
        train_config=TrainConfig(
            scale=2,
            batch_size=2,
            total_steps=3,
            save_every=1,
            log_every=1,
            device="cpu",
        ),
        synthetic_count=32,
        resume_path=checkpoint_path,
    )
    resumed_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    assert resumed_checkpoint["step"] == 3


def test_training_exports_inference_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "demo.pt"
    train(
        train_dir=None,
        output_path=checkpoint_path,
        model_config=model_config_from_variant("tiny", upscale=2),
        train_config=TrainConfig(
            scale=2,
            batch_size=2,
            total_steps=1,
            save_every=1,
            log_every=1,
            device="cpu",
        ),
        synthetic_count=16,
    )
    inference_checkpoint = checkpoint_path.with_name("demo_inference.pt")
    assert inference_checkpoint.exists()
    payload = torch.load(inference_checkpoint, map_location="cpu")
    assert payload["checkpoint_kind"] == "inference"
    assert "ema_state" in payload


def test_auto_device_context_prefers_available_accelerator() -> None:
    context = build_device_context("auto", amp_mode="off", channels_last=True)
    assert context.device.type in {"cpu", "cuda", "mps"}
    if context.device.type == "cpu":
        assert context.channels_last is False
