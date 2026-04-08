from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epnet import EPNet  # noqa: E402
from epnet.config import ModelConfig, model_config_from_variant  # noqa: E402
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


def test_epnet_supports_x2_x3_and_x4_shapes_without_nans() -> None:
    input_tensor = torch.randn(1, 3, 13, 17)
    for scale in [2, 3, 4]:
        model = EPNet(ModelConfig(upscale=scale)).eval()
        with torch.no_grad():
            output = model(input_tensor)
        assert output.shape == (1, 3, 13 * scale, 17 * scale)
        assert not torch.isnan(output).any()


def test_tiny_variant_is_smaller_than_paper_variant() -> None:
    tiny = EPNet(model_config_from_variant("tiny"))
    paper = EPNet(model_config_from_variant("paper"))
    assert count_parameters(tiny) < count_parameters(paper)


def test_invalid_model_config_values_fail_clearly() -> None:
    with pytest.raises(ValueError, match="upscale"):
        ModelConfig(upscale=1)
    with pytest.raises(ValueError, match="num_pfem"):
        ModelConfig(num_pfem=0)
    with pytest.raises(ValueError, match="espm_levels"):
        ModelConfig(espm_levels=1)
    with pytest.raises(ValueError, match="split_ratio"):
        ModelConfig(split_ratio=1.2)
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(embed_dim=40, num_heads=3)
