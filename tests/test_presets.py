# ruff: noqa: I001, E402
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epnet.models.presets import available_model_presets, build_model_preset  # noqa: E402
from epnet.profiling import count_parameters  # noqa: E402
from epnet import EPNet  # noqa: E402


def test_expected_model_presets_exist() -> None:
    presets = available_model_presets()
    assert "paper_like" in presets
    assert "edge_tiny" in presets
    assert "edge_default" in presets
    assert "balanced_quality" in presets


def test_edge_default_is_lighter_than_paper_like() -> None:
    edge_default = EPNet(build_model_preset("edge_default"))
    paper_like = EPNet(build_model_preset("paper_like"))
    assert count_parameters(edge_default) < count_parameters(paper_like)
