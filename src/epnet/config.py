from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict


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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InferenceConfig:
    device: str = "cpu"
    clamp_output: bool = True
    tile_size: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
