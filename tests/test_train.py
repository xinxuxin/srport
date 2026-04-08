from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from epnet import EPNet  # noqa: E402
from epnet.config import ModelConfig, TrainConfig, model_config_from_variant  # noqa: E402
from epnet.device import build_device_context  # noqa: E402
from epnet.metrics import evaluate_prediction  # noqa: E402
from epnet.model import load_checkpoint, load_model_from_checkpoint  # noqa: E402
from epnet.train import _retain_checkpoint_history, train, validate  # noqa: E402
from epnet.utils import pil_to_tensor  # noqa: E402


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


def test_checkpoint_load_after_training_supports_forward(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "forward.pt"
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
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    model = load_model_from_checkpoint(checkpoint).eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 12, 12))
    assert output.shape == (1, 3, 24, 24)


def test_validation_smoke_runs_on_tiny_image_folder(tmp_path: Path) -> None:
    hr_dir = tmp_path / "hr"
    hr_dir.mkdir(parents=True)
    image = Image.new("RGB", (32, 32), color=(120, 80, 200))
    image.save(hr_dir / "sample.png")
    model = EPNet(model_config_from_variant("tiny", upscale=2)).eval()
    metrics = validate(
        model,
        hr_dir,
        scale=2,
        device_context=build_device_context("cpu", amp_mode="off", channels_last=False),
        max_images=1,
    )
    assert metrics["samples"] == 1.0
    assert "psnr" in metrics
    assert "ssim" in metrics


def test_evaluate_prediction_rejects_too_large_shave() -> None:
    image = pil_to_tensor(Image.new("RGB", (8, 8), color=(0, 0, 0)))
    with pytest.raises(ValueError, match="Shave"):
        evaluate_prediction(image, image, shave=5)


def test_missing_checkpoint_path_fails_clearly(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pt"
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        load_checkpoint(missing)


def test_checkpoint_history_retains_latest_numeric_steps(tmp_path: Path) -> None:
    for step in (800, 900, 1000):
        (tmp_path / f"step{step}.pt").write_text(str(step), encoding="utf-8")
    _retain_checkpoint_history(tmp_path / "latest.pt", keep=2)
    remaining = sorted(path.name for path in tmp_path.glob("step*.pt"))
    assert remaining == ["step1000.pt", "step900.pt"]
