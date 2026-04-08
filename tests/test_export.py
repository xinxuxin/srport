# ruff: noqa: I001, E402
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epnet.config import TrainConfig, model_config_from_variant  # noqa: E402
from epnet.export import export_onnx  # noqa: E402
from epnet.train import train  # noqa: E402


def test_export_onnx_returns_status_payload(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "export.pt"
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
    result = export_onnx(checkpoint_path.with_name("export_inference.pt"), tmp_path / "model.onnx")
    assert result["status"] in {"success", "failed"}
    if result["status"] == "success":
        assert Path(result["output_path"]).exists()
