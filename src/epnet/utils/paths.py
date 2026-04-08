from __future__ import annotations

from pathlib import Path


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_outputs_root() -> Path:
    return project_root() / "outputs"


def default_data_root() -> Path:
    return project_root() / "data"
