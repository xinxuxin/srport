from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from .config import ModelConfig
from .modules import ESPM, PFEMSubmodule


class EPNet(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.shallow = nn.Conv2d(
            self.config.in_channels,
            self.config.embed_dim,
            kernel_size=3,
            padding=1,
        )
        if self.config.share_pfem_weights:
            shared = PFEMSubmodule(
                self.config.embed_dim,
                self.config.num_heads,
                self.config.window_size,
                self.config.mlp_ratio,
            )
            self.pfem_blocks = nn.ModuleList(shared for _ in range(self.config.num_pfem))
        else:
            self.pfem_blocks = nn.ModuleList(
                PFEMSubmodule(
                    self.config.embed_dim,
                    self.config.num_heads,
                    self.config.window_size,
                    self.config.mlp_ratio,
                )
                for _ in range(self.config.num_pfem)
            )
        self.espm = ESPM(
            self.config.embed_dim,
            self.config.espm_levels,
            self.config.split_ratio,
        )
        self.reconstruction = nn.Sequential(
            nn.Conv2d(
                self.config.embed_dim,
                self.config.in_channels * (self.config.upscale**2),
                kernel_size=3,
                padding=1,
            ),
            nn.PixelShuffle(self.config.upscale),
        )

    def forward_features(self, x: Tensor) -> tuple[Tensor, Tensor]:
        base = self.shallow(x)
        pfem = base
        for block in self.pfem_blocks:
            pfem = block(pfem)
        espm = self.espm(base)
        return pfem, espm

    def forward(self, x: Tensor) -> Tensor:
        pfem, espm = self.forward_features(x)
        return self.reconstruction(pfem + espm)

    def profile_input_shape(self) -> list[int]:
        return [1, self.config.in_channels, 48, 48]


def load_model_from_checkpoint(checkpoint: dict[str, object]) -> EPNet:
    raw_config = checkpoint.get("model_config", {})
    config = ModelConfig(**raw_config) if isinstance(raw_config, dict) else ModelConfig()
    model = EPNet(config)
    state = checkpoint.get("model_state")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint is missing model_state")
    model.load_state_dict(state)
    return model


def load_checkpoint(
    checkpoint_path: Path | str,
    map_location: str | torch.device = "cpu",
) -> dict[str, object]:
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must deserialize to a dictionary.")
    return checkpoint
