# ruff: noqa: I001, E402
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epnet.experiments import (  # noqa: E402
    ExperimentSpec,
    create_synthetic_eval_folder,
    render_markdown_report,
    run_experiment,
)


def test_create_synthetic_eval_folder_writes_expected_images(tmp_path: Path) -> None:
    output_dir = create_synthetic_eval_folder(tmp_path / "eval", image_count=3, size=64, seed=7)
    images = sorted(output_dir.glob("*.png"))
    assert len(images) == 3


def test_run_experiment_returns_profile_and_eval(tmp_path: Path) -> None:
    eval_dir = create_synthetic_eval_folder(tmp_path / "eval", image_count=2, size=64, seed=11)
    result = run_experiment(
        ExperimentSpec(
            name="tiny_cpu_smoke",
            group="variant",
            variant="tiny",
            description="CPU smoke run for experiment pipeline.",
            overrides={},
        ),
        output_dir=tmp_path,
        scale=2,
        steps=1,
        synthetic_count=16,
        device="cpu",
        val_dir=eval_dir,
    )
    assert Path(result["final_checkpoint"]).exists()
    assert Path(result["inference_checkpoint"]).exists()
    assert result["evaluation"]["samples"] == 2
    assert result["profile"]["parameters"] > 0


def test_render_markdown_report_includes_summary_sections() -> None:
    markdown = render_markdown_report(
        [
            {
                "name": "variant_tiny",
                "group": "variant",
                "variant": "tiny",
                "description": "Tiny preset.",
                "evaluation": {"average_psnr": 1.0, "average_ssim": 0.1},
                "profile": {"parameters": 10, "macs": 20, "latency_ms": 3.0},
                "best_psnr": 1.0,
                "train_wall_time_s": 0.5,
            },
            {
                "name": "ablation_espm2",
                "group": "ablation",
                "variant": "paper",
                "description": "Reduced ESPM depth.",
                "evaluation": {"average_psnr": 0.9, "average_ssim": 0.09},
                "profile": {"parameters": 12, "macs": 22, "latency_ms": 3.5},
                "best_psnr": 0.9,
                "train_wall_time_s": 0.6,
            },
        ],
        scale=2,
        steps=8,
        synthetic_count=128,
        device="mps",
        eval_dir=Path("/tmp/eval"),
    )
    assert "Variant Comparison" in markdown
    assert "Ablation Comparison" in markdown
    assert "Key Takeaways" in markdown
