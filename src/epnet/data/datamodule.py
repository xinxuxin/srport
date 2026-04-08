from __future__ import annotations

from pathlib import Path

from torch.utils.data import Dataset

from ..config import DataConfig, TrainConfig
from .benchmark import EvaluationImageDataset
from .div2k import benchmark_hr_dir, validate_real_data_paths
from .paired_dataset import TrainImageFolderDataset
from .synthetic import SyntheticPatternDataset


def build_train_dataset(
    data_config: DataConfig,
    train_config: TrainConfig,
) -> Dataset[tuple]:
    if data_config.dataset_type == "synthetic":
        return SyntheticPatternDataset(
            count=max(128, data_config.synthetic_count or train_config.batch_size * 8),
            hr_size=train_config.patch_size,
            scale=train_config.scale,
            seed=train_config.seed,
        )

    validate_real_data_paths(Path(data_config.dataset_root))
    return TrainImageFolderDataset(
        Path(data_config.train_hr_dir),
        train_config.patch_size,
        train_config.scale,
    )


def build_validation_datasets(
    data_config: DataConfig,
    scale: int,
) -> dict[str, EvaluationImageDataset]:
    if data_config.dataset_type == "synthetic":
        eval_dir = Path(data_config.processed_root) / f"synthetic_eval_x{scale}"
        return {"synthetic_eval": EvaluationImageDataset(eval_dir, scale)}

    dataset_root = Path(data_config.dataset_root)
    datasets: dict[str, EvaluationImageDataset] = {}
    for name in ("Set5", "Set14", "BSD100", "Urban100"):
        hr_dir = Path(data_config.benchmark_roots.get(name, benchmark_hr_dir(dataset_root, name)))
        if hr_dir.exists():
            datasets[name] = EvaluationImageDataset(hr_dir, scale)
    if not datasets:
        fallback = Path(data_config.train_hr_dir)
        datasets["DIV2K_train_subset"] = EvaluationImageDataset(fallback, scale)
    return datasets
