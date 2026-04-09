"""Helpers for named EPNet presets.

This module is intentionally thin. The canonical preset values live in
``epnet.config`` so training and deployment use the same source of truth, while
this file provides a friendlier model-layer API for building presets by name.
"""

from __future__ import annotations

from typing import Any

from ..config import MODEL_PRESETS, ModelConfig, model_config_from_variant, normalize_model_preset


def available_model_presets() -> list[str]:
    """Return the preset names that should be shown in CLIs and docs."""
    return sorted(MODEL_PRESETS)


def build_model_preset(name: str, **overrides: Any) -> ModelConfig:
    """Build a validated ``ModelConfig`` from a preset name plus overrides."""
    return model_config_from_variant(normalize_model_preset(name), **overrides)
