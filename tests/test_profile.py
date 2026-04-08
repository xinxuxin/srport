# ruff: noqa: I001, E402
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epnet.config import TrainConfig, model_config_from_variant  # noqa: E402
from epnet.profile import profile_checkpoint  # noqa: E402
from epnet.train import train  # noqa: E402


def test_profile_checkpoint_reports_extended_metrics(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "profile.pt"
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
    payload = profile_checkpoint(
        checkpoint_path.with_name("profile_inference.pt"),
        device="cpu",
        output_json=tmp_path / "profile.json",
        output_markdown=tmp_path / "profile.md",
    )
    assert payload["checkpoint_size_bytes"] > 0
    assert payload["model_state_size_bytes"] > 0
    assert payload["estimated_memory_bytes"] >= payload["model_state_size_bytes"]
