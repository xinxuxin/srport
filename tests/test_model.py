from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epnet import EPNet  # noqa: E402
from epnet.profiling import count_parameters  # noqa: E402


def test_epnet_upscales_by_factor_four() -> None:
    model = EPNet().eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 12, 12))
    assert output.shape == (1, 3, 48, 48)


def test_epnet_parameter_regime_is_lightweight() -> None:
    model = EPNet()
    params = count_parameters(model)
    assert 430_000 <= params <= 520_000
