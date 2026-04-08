from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epnet import EPNet  # noqa: E402
from epnet.profiling import profile_model  # noqa: E402


def test_profile_model_reports_stable_macs_across_iterations() -> None:
    model = EPNet().eval()
    input_tensor = torch.randn(1, 3, 16, 16)
    once = profile_model(model, input_tensor, warmup=0, iters=1)
    many = profile_model(model, input_tensor, warmup=0, iters=5)
    assert once.macs == many.macs
    assert once.flops == many.flops


def test_profile_model_rejects_invalid_iteration_settings() -> None:
    model = EPNet().eval()
    input_tensor = torch.randn(1, 3, 16, 16)
    try:
        profile_model(model, input_tensor, warmup=-1, iters=1)
    except ValueError as error:
        assert "warmup" in str(error)
    else:
        raise AssertionError("Expected invalid warmup to raise ValueError.")
