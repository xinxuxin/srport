from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SUPPORTED_BENCHMARKS = ("Set5", "Set14", "BSD100", "Urban100", "Manga109")


@dataclass(frozen=True)
class DatasetPaths:
    root: Path
    raw_root: Path
    processed_root: Path
    benchmarks_root: Path
    div2k_train_hr: Path
    div2k_valid_hr: Path


def build_dataset_paths(dataset_root: Path) -> DatasetPaths:
    raw_root = dataset_root / "raw"
    processed_root = dataset_root / "processed"
    benchmarks_root = dataset_root / "benchmarks"
    div2k_root = raw_root / "DIV2K"
    return DatasetPaths(
        root=dataset_root,
        raw_root=raw_root,
        processed_root=processed_root,
        benchmarks_root=benchmarks_root,
        div2k_train_hr=div2k_root / "DIV2K_train_HR",
        div2k_valid_hr=div2k_root / "DIV2K_valid_HR",
    )


def benchmark_hr_dir(dataset_root: Path, name: str) -> Path:
    if name not in SUPPORTED_BENCHMARKS:
        expected = ", ".join(SUPPORTED_BENCHMARKS)
        raise ValueError(f"Unsupported benchmark '{name}'. Expected one of: {expected}.")
    return dataset_root / "benchmarks" / name / "HR"


def validate_real_data_paths(dataset_root: Path) -> dict[str, Path]:
    paths = build_dataset_paths(dataset_root)
    required = {
        "div2k_train_hr": paths.div2k_train_hr,
        "div2k_valid_hr": paths.div2k_valid_hr,
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        formatted = ", ".join(missing)
        raise FileNotFoundError(
            "Missing required real-data directories: "
            f"{formatted}. Run the dataset setup scripts or place the datasets manually."
        )
    return required
