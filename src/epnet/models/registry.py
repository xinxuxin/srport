from __future__ import annotations

from typing import Any

from ..config import ModelConfig
from .epnet import EPNet
from .presets import build_model_preset


def build_model(config: ModelConfig) -> EPNet:
    return EPNet(config)


def build_model_from_preset(name: str, **overrides: Any) -> EPNet:
    return EPNet(build_model_preset(name, **overrides))
