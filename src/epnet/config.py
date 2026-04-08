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

    def __post_init__(self) -> None:
        if self.upscale not in {2, 3, 4}:
            raise ValueError("upscale must be one of: 2, 3, 4.")
        if self.in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        if self.embed_dim <= 0:
            raise ValueError("embed_dim must be positive.")
        if self.num_pfem <= 0:
            raise ValueError("num_pfem must be >= 1.")
        if self.window_size <= 0:
            raise ValueError("window_size must be positive.")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if self.embed_dim % self.num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive.")
        if self.espm_levels < 2:
            raise ValueError("espm_levels must be >= 2.")
        if not 0.0 < self.split_ratio < 1.0:
            raise ValueError("split_ratio must be between 0 and 1.")

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

    def __post_init__(self) -> None:
        if self.scale not in {2, 3, 4}:
            raise ValueError("scale must be one of: 2, 3, 4.")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if not 0.0 <= self.beta1 < 1.0 or not 0.0 <= self.beta2 < 1.0:
            raise ValueError("Adam betas must be in [0, 1).")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1).")
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive.")
        if self.save_every <= 0:
            raise ValueError("save_every must be positive.")
        if self.log_every <= 0:
            raise ValueError("log_every must be positive.")
        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0.")
        if self.grad_clip_norm < 0:
            raise ValueError("grad_clip_norm must be >= 0.")
        if self.val_every < 0:
            raise ValueError("val_every must be >= 0.")
        if self.max_validation_images <= 0:
            raise ValueError("max_validation_images must be positive.")
        if self.checkpoint_history < 0:
            raise ValueError("checkpoint_history must be >= 0.")

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
