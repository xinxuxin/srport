"""Small path helpers for repository-relative defaults."""

from __future__ import annotations

from pathlib import Path


def ensure_dir(path: Path) -> None:
    """Create a directory tree if it does not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def project_root() -> Path:
    """Return the repository root based on this file location."""
    return Path(__file__).resolve().parents[3]


def default_outputs_root() -> Path:
    """Return the default outputs directory under the repository root."""
    return project_root() / "outputs"


def default_data_root() -> Path:
    """Return the default local data directory under the repository root."""
    return project_root() / "data"
