"""Small model-construction registry for the EPNet codebase.

The project currently exposes one model family, but keeping creation helpers in
one place makes training, ablation, and future extensions easier to follow.
"""

from __future__ import annotations

from typing import Any

from ..config import ModelConfig
from .epnet import EPNet
from .presets import build_model_preset


def build_model(config: ModelConfig) -> EPNet:
    """Construct EPNet from an already validated configuration object."""
    return EPNet(config)


def build_model_from_preset(name: str, **overrides: Any) -> EPNet:
    """Construct EPNet directly from a named preset."""
    return EPNet(build_model_preset(name, **overrides))
