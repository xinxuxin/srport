# ruff: noqa: I001, E402
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epnet.config import (  # noqa: E402
    DataConfig,
    TrainConfig,
    load_data_config,
    load_train_config,
    model_config_from_variant,
)
from epnet.train import run_training_from_configs  # noqa: E402


def test_config_parsing_for_train_and_eval_paths() -> None:
    data_config = load_data_config(ROOT / "configs/data/synthetic_x2.yaml")
    train_config = load_train_config(ROOT / "configs/train/smoke.yaml")
    assert data_config.dataset_type == "synthetic"
    assert train_config.scale == 2


def test_config_driven_synthetic_smoke_run(tmp_path: Path) -> None:
    run_dir = run_training_from_configs(
        model_config=model_config_from_variant("edge_default", upscale=2),
        data_config=DataConfig(
            dataset_type="synthetic",
            dataset_root=str(tmp_path / "data"),
            processed_root=str(tmp_path / "processed"),
            scale=2,
            synthetic_count=16,
            synthetic_eval_count=2,
            synthetic_image_size=64,
        ),
        train_config=TrainConfig(
            scale=2,
            batch_size=2,
            total_steps=2,
            save_every=1,
            log_every=1,
            val_every=1,
            max_validation_images=2,
            device="cpu",
            amp="off",
            run_name="smoke_run",
            output_root=str(tmp_path / "outputs"),
            auto_resume=False,
            checkpoint_history=1,
        ),
    )
    assert (run_dir / "latest.pt").exists()
    assert (run_dir / "best.pt").exists()
    assert (run_dir / "eval.json").exists()
    assert (run_dir / "profile.json").exists()
