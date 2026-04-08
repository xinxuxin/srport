from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any

MODEL_VARIANTS: dict[str, dict[str, Any]] = {
    "tiny": {
        "embed_dim": 32,
        "num_pfem": 3,
        "num_heads": 4,
        "mlp_ratio": 2.0,
        "espm_levels": 3,
    },
    "paper": {
        "embed_dim": 40,
        "num_pfem": 4,
        "num_heads": 4,
        "mlp_ratio": 2.0,
        "espm_levels": 3,
    },
    "balanced": {
        "embed_dim": 48,
        "num_pfem": 4,
        "num_heads": 6,
        "mlp_ratio": 2.0,
        "espm_levels": 3,
    },
}


@dataclass(frozen=True)
class ModelConfig:
    upscale: int = 4
    in_channels: int = 3
    embed_dim: int = 40
    num_pfem: int = 4
    window_size: int = 8
    num_heads: int = 4
    mlp_ratio: float = 2.0
    espm_levels: int = 3
    split_ratio: float = 0.5
    share_pfem_weights: bool = False
    variant: str = "paper"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def model_config_from_variant(variant: str = "paper", **overrides: Any) -> ModelConfig:
    normalized = variant.lower()
    if normalized not in MODEL_VARIANTS:
        expected = ", ".join(sorted(MODEL_VARIANTS))
        raise ValueError(
            f"Unknown model variant '{variant}'. Expected one of: {expected}."
        )
    return ModelConfig(**MODEL_VARIANTS[normalized], variant=normalized, **overrides)


@dataclass(frozen=True)
class TrainConfig:
    scale: int = 4
    patch_size: int = 48
    batch_size: int = 32
    learning_rate: float = 5e-4
    beta1: float = 0.9
    beta2: float = 0.99
    weight_decay: float = 0.0
    ema_decay: float = 0.999
    total_steps: int = 1_000_000
    save_every: int = 1_000
    log_every: int = 50
    num_workers: int = 0
    seed: int = 42
    device: str = "auto"
    amp: str = "auto"
    grad_clip_norm: float = 0.0
    channels_last: bool = True
    compile_model: bool = False
    val_every: int = 0
    max_validation_images: int = 8
    save_best: bool = True
    checkpoint_history: int = 2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def replace(self, **changes: Any) -> TrainConfig:
        return replace(self, **changes)


@dataclass(frozen=True)
class InferenceConfig:
    device: str = "cpu"
    clamp_output: bool = True
    tile_size: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
