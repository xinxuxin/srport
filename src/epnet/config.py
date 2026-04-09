"""Configuration dataclasses and YAML loaders for EPNet training/inference.

This module is shared by the research-facing training pipeline and the
deployment-facing checkpoint export/evaluation path. It centralizes the small
set of configuration objects that define:

- which EPNet preset to instantiate
- which dataset family to read
- which training schedule to run
- which inference-time safeguards to apply

The paper provides high-level defaults such as scale, patch size, and optimizer
settings, but it does not define a product-ready configuration system. The
dataclasses here are therefore an engineering layer that turns the paper ideas
into reproducible, typed, and validated runtime settings.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

LEGACY_PRESET_ALIASES = {
    "tiny": "edge_tiny",
    "paper": "paper_like",
    "balanced": "balanced_quality",
}

MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "paper_like": {
        "embed_dim": 40,
        "num_pfem": 4,
        "num_heads": 4,
        "mlp_ratio": 2.0,
        "espm_levels": 3,
        "global_context": "swin",
    },
    "edge_tiny": {
        "embed_dim": 32,
        "num_pfem": 3,
        "num_heads": 4,
        "mlp_ratio": 2.0,
        "espm_levels": 3,
        "global_context": "swin",
    },
    "edge_default": {
        "embed_dim": 40,
        "num_pfem": 2,
        "num_heads": 4,
        "mlp_ratio": 2.0,
        "espm_levels": 2,
        "global_context": "swin",
    },
    "balanced_quality": {
        "embed_dim": 48,
        "num_pfem": 4,
        "num_heads": 6,
        "mlp_ratio": 2.0,
        "espm_levels": 3,
        "global_context": "swin",
    },
}


@dataclass(frozen=True)
class ModelConfig:
    """Typed description of the EPNet architecture used for one run.

    Summary:
        Captures the high-level network shape, including the upscale factor,
        channel width, PFEM depth, ESPM depth, and global-context block type.

    Paper mapping:
        This config corresponds to the architectural degrees of freedom around
        the paper's shallow stem, PFEM stages, ESPM branch, and reconstruction
        head.

    Role in the system:
        The training loop, evaluation path, profiling code, and deployment
        artifact loader all rely on this config to rebuild the same EPNet
        instance from a checkpoint.
    """

    upscale: int = 4
    in_channels: int = 3
    embed_dim: int = 40
    num_pfem: int = 2
    window_size: int = 8
    num_heads: int = 4
    mlp_ratio: float = 2.0
    espm_levels: int = 2
    split_ratio: float = 0.5
    share_pfem_weights: bool = False
    global_context: str = "swin"
    variant: str = "edge_default"

    def __post_init__(self) -> None:
        """Validate core architectural invariants before model construction."""
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
        if self.global_context not in {"swin", "lightweight_conv_context", "none"}:
            raise ValueError(
                "global_context must be one of: swin, lightweight_conv_context, none."
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-friendly representation used in manifests."""
        return asdict(self)


@dataclass(frozen=True)
class DataConfig:
    """Dataset configuration for either real-data or synthetic workflows.

    Summary:
        Describes where HR images live, where bicubic LR caches are written,
        which benchmark datasets should be evaluated, and how synthetic smoke
        data should be generated when the repository runs in regression mode.

    Role in the system:
        The data module uses this config to decide whether it should build the
        real DIV2K training path or the synthetic smoke path.
    """

    dataset_type: str = "div2k"
    dataset_root: str = "data"
    train_hr_dir: str = "data/raw/DIV2K/DIV2K_train_HR"
    benchmark_roots: dict[str, str] = field(
        default_factory=lambda: {
            "Set5": "data/benchmarks/Set5/HR",
            "Set14": "data/benchmarks/Set14/HR",
            "BSD100": "data/benchmarks/BSD100/HR",
            "Urban100": "data/benchmarks/Urban100/HR",
        }
    )
    processed_root: str = "data/processed"
    cache_lr: bool = True
    scale: int = 4
    synthetic_count: int = 256
    synthetic_eval_count: int = 6
    synthetic_image_size: int = 96

    def __post_init__(self) -> None:
        """Validate dataset family and key size/count assumptions."""
        if self.dataset_type not in {"div2k", "synthetic"}:
            raise ValueError("dataset_type must be one of: div2k, synthetic.")
        if self.scale not in {2, 3, 4}:
            raise ValueError("scale must be one of: 2, 3, 4.")
        if self.synthetic_count <= 0:
            raise ValueError("synthetic_count must be positive.")
        if self.synthetic_eval_count <= 0:
            raise ValueError("synthetic_eval_count must be positive.")
        if self.synthetic_image_size <= 0:
            raise ValueError("synthetic_image_size must be positive.")

    def to_dict(self) -> dict[str, Any]:
        """Return a serializable snapshot for manifests and debug output."""
        return asdict(self)


@dataclass(frozen=True)
class TrainConfig:
    """Training schedule and runtime policy for one experiment run.

    Summary:
        Encodes optimizer hyperparameters, checkpoint cadence, validation
        cadence, device/AMP policy, and output directory naming.

    Paper mapping:
        Fields such as ``patch_size``, ``batch_size``, ``learning_rate``, and
        ``ema_decay`` are the code-level representation of the paper's reported
        training defaults.

    Implementation notes:
        This config also carries product-oriented choices that are not part of
        the paper itself, such as auto-resume, checkpoint history retention,
        and manifest generation.
    """

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
    val_every: int = 1_000
    max_validation_images: int = 8
    save_best: bool = True
    checkpoint_history: int = 2
    run_name: str = "run_x4_edge_default"
    output_root: str = "outputs"
    auto_resume: bool = True
    manifest_command: str = ""

    def __post_init__(self) -> None:
        """Reject invalid schedules early so runs fail before allocating data."""
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
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: auto, cpu, cuda, mps.")
        if self.amp not in {"auto", "on", "off"}:
            raise ValueError("amp must be one of: auto, on, off.")

    def to_dict(self) -> dict[str, Any]:
        """Return a manifest-friendly representation."""
        return asdict(self)

    def replace(self, **changes: Any) -> TrainConfig:
        """Create a modified copy without mutating the frozen dataclass."""
        return replace(self, **changes)


@dataclass(frozen=True)
class InferenceConfig:
    """Lightweight runtime options for direct checkpoint inference utilities."""

    device: str = "cpu"
    clamp_output: bool = True
    tile_size: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return a manifest/debug representation."""
        return asdict(self)


def normalize_model_preset(name: str) -> str:
    """Map legacy preset aliases onto the current naming scheme."""
    normalized = name.lower()
    return LEGACY_PRESET_ALIASES.get(normalized, normalized)


def model_config_from_variant(variant: str = "edge_default", **overrides: Any) -> ModelConfig:
    """Build a :class:`ModelConfig` from a named preset plus explicit overrides.

    This helper is the bridge between human-friendly preset names such as
    ``edge_default`` and the fully expanded dataclass consumed by the model
    registry.
    """
    normalized = normalize_model_preset(variant)
    if normalized not in MODEL_PRESETS:
        expected = ", ".join(sorted(MODEL_PRESETS))
        raise ValueError(
            f"Unknown model variant '{variant}'. Expected one of: {expected}."
        )
    return ModelConfig(**MODEL_PRESETS[normalized], variant=normalized, **overrides)


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML config file and require the top level to be a mapping."""
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return payload


def load_model_config(path: Path) -> ModelConfig:
    """Load a model config YAML and expand its ``preset`` into full values."""
    payload = _load_yaml(path)
    preset = payload.pop("preset", payload.pop("variant", "edge_default"))
    return model_config_from_variant(str(preset), **payload)


def load_data_config(path: Path) -> DataConfig:
    """Load a data config YAML into a validated :class:`DataConfig`."""
    return DataConfig(**_load_yaml(path))


def load_train_config(path: Path) -> TrainConfig:
    """Load a training config YAML into a validated :class:`TrainConfig`."""
    return TrainConfig(**_load_yaml(path))
