# ruff: noqa: E402
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epnet.config import load_data_config, load_model_config, load_train_config
from epnet.data import validate_real_data_paths
from epnet.train import run_training_from_configs
from epnet.utils import build_device_context


def main() -> None:
    model_config_path = ROOT / "configs/model/edge_default.yaml"
    data_config_path = ROOT / "configs/data/div2k_x4.yaml"
    train_config_path = ROOT / "configs/train/local_full_x4.yaml"
    model_config = load_model_config(model_config_path)
    data_config = load_data_config(data_config_path)
    train_config = load_train_config(train_config_path)
    validate_real_data_paths(Path(data_config.dataset_root))
    device_context = build_device_context(
        train_config.device,
        train_config.amp,
        train_config.channels_last,
    )
    run_dir = Path(train_config.output_root) / train_config.run_name
    latest_checkpoint = run_dir / "latest.pt"
    print(f"Starting local full x4 training on device: {device_context.device.type}")
    print(f"Model config: {model_config_path}")
    print(f"Data config: {data_config_path}")
    print(f"Train config: {train_config_path}")
    print(f"Output directory: {run_dir}")
    print(f"Auto-resume checkpoint: {latest_checkpoint}")
    run_dir = run_training_from_configs(
        model_config=model_config,
        data_config=data_config,
        train_config=train_config.replace(
            manifest_command="python scripts/run_local_full_x4.py"
        ),
    )
    print(f"Completed run in {run_dir}")


if __name__ == "__main__":
    main()
