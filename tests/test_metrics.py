from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epnet.metrics import calculate_psnr, calculate_ssim  # noqa: E402


def test_psnr_is_infinite_for_identical_images() -> None:
    image = np.zeros((8, 8), dtype=np.float32)
    assert calculate_psnr(image, image) == float("inf")


def test_ssim_is_near_one_for_identical_images() -> None:
    image = np.full((16, 16), 127.0, dtype=np.float32)
    score = calculate_ssim(image, image)
    assert 0.99 <= score <= 1.0
