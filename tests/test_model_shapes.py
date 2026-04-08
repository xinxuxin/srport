# ruff: noqa: I001, E402
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epnet import EPNet  # noqa: E402
from epnet.config import ModelConfig  # noqa: E402


def test_edge_default_forward_sanity() -> None:
    model = EPNet().eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 16, 16))
    assert output.shape == (1, 3, 64, 64)


def test_x2_x3_x4_output_shapes_are_exact() -> None:
    input_tensor = torch.randn(1, 3, 15, 19)
    for scale in (2, 3, 4):
        model = EPNet(ModelConfig(upscale=scale)).eval()
        with torch.no_grad():
            output = model(input_tensor)
        assert output.shape == (1, 3, 15 * scale, 19 * scale)
