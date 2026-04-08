from __future__ import annotations

from typing import Any

from ..config import MODEL_PRESETS, ModelConfig, model_config_from_variant, normalize_model_preset


def available_model_presets() -> list[str]:
    return sorted(MODEL_PRESETS)


def build_model_preset(name: str, **overrides: Any) -> ModelConfig:
    return model_config_from_variant(normalize_model_preset(name), **overrides)
