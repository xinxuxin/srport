# ruff: noqa: I001, E402
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epnet.config import TrainConfig, model_config_from_variant  # noqa: E402
from epnet.evaluate import evaluate  # noqa: E402
from epnet.train import train  # noqa: E402


def test_eval_smoke_returns_metrics(tmp_path: Path) -> None:
    hr_dir = tmp_path / "hr"
    hr_dir.mkdir(parents=True)
    Image.new("RGB", (32, 32), color=(50, 120, 180)).save(hr_dir / "sample.png")
    checkpoint_path = tmp_path / "eval.pt"
    train(
        train_dir=None,
        output_path=checkpoint_path,
        model_config=model_config_from_variant("edge_tiny", upscale=2),
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
    summary = evaluate(checkpoint_path.with_name("eval_inference.pt"), hr_dir, device="cpu")
    assert summary["samples"] == 1
    assert "average_psnr" in summary
    assert "average_ssim" in summary
