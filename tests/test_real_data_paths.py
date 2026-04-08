# ruff: noqa: I001, E402
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epnet.data import build_dataset_paths, validate_real_data_paths  # noqa: E402


def test_build_dataset_paths_returns_expected_layout(tmp_path: Path) -> None:
    paths = build_dataset_paths(tmp_path)
    assert paths.div2k_train_hr == tmp_path / "raw" / "DIV2K" / "DIV2K_train_HR"
    assert paths.div2k_valid_hr == tmp_path / "raw" / "DIV2K" / "DIV2K_valid_HR"


def test_validate_real_data_paths_fails_clearly_when_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Missing required real-data directories"):
        validate_real_data_paths(tmp_path)
