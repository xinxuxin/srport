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
    model_config = load_model_config(ROOT / "configs/model/edge_default.yaml")
    model_config = type(model_config)(**{**model_config.to_dict(), "upscale": 2})
    data_config = load_data_config(ROOT / "configs/data/div2k_x2.yaml")
    train_config = load_train_config(ROOT / "configs/train/local_full_x2.yaml")
    validate_real_data_paths(Path(data_config.dataset_root))
    device_context = build_device_context(
        train_config.device,
        train_config.amp,
        train_config.channels_last,
    )
    print(f"Starting local full x2 training on device: {device_context.device.type}")
    print(f"Run name: {train_config.run_name}")
    run_dir = run_training_from_configs(
        model_config=model_config,
        data_config=data_config,
        train_config=train_config.replace(
            manifest_command="python scripts/run_local_full_x2.py"
        ),
    )
    print(f"Completed run in {run_dir}")


if __name__ == "__main__":
    main()
