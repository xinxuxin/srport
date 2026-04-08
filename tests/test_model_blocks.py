from __future__ import annotations
# ruff: noqa: I001, E402

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epnet.modules import DynamicChannelAttentionBlock, ESAB, ESPM, LFEB, PFEMSubmodule, SwinBlock  # noqa: E402


def test_lfeb_forward_preserves_shape_without_nans() -> None:
    module = LFEB(40).eval()
    x = torch.randn(1, 40, 13, 17)
    with torch.no_grad():
        y = module(x)
    assert y.shape == x.shape
    assert not torch.isnan(y).any()


def test_esab_forward_preserves_shape_without_nans() -> None:
    module = ESAB(40).eval()
    x = torch.randn(1, 40, 13, 17)
    with torch.no_grad():
        y = module(x)
    assert y.shape == x.shape
    assert not torch.isnan(y).any()


def test_dcab_forward_preserves_shape_without_nans() -> None:
    module = DynamicChannelAttentionBlock(40, 0.5).eval()
    x = torch.randn(1, 40, 13, 17)
    with torch.no_grad():
        y = module(x)
    assert y.shape == x.shape
    assert not torch.isnan(y).any()


def test_espm_forward_preserves_shape_without_nans() -> None:
    module = ESPM(40, 3, 0.5).eval()
    x = torch.randn(1, 40, 13, 17)
    with torch.no_grad():
        y = module(x)
    assert y.shape == x.shape
    assert not torch.isnan(y).any()


def test_pfem_submodule_forward_preserves_shape_without_nans() -> None:
    module = PFEMSubmodule(40, 4, 8, 2.0).eval()
    x = torch.randn(1, 40, 13, 17)
    with torch.no_grad():
        y = module(x)
    assert y.shape == x.shape
    assert not torch.isnan(y).any()


def test_swin_block_forward_preserves_shape_without_nans() -> None:
    module = SwinBlock(40, 4, 8, 4, 2.0).eval()
    x = torch.randn(1, 40, 13, 17)
    with torch.no_grad():
        y = module(x)
    assert y.shape == x.shape
    assert not torch.isnan(y).any()
