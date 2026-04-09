"""Top-level EPNet architecture used by training, evaluation, and deployment.

This file contains the main model class that stitches together the paper-inspired
EPNet structure:

- a shallow 3x3 convolution for feature projection
- a repeated PFEM pathway for detail-oriented feature refinement
- an ESPM branch for efficient multiscale context aggregation
- a reconstruction head that upsamples with PixelShuffle

The implementation is intentionally compact and deployment-friendly. It aims to
preserve the architectural spirit of the paper while remaining easy to train,
profile, export, and serve inside this repository's end-to-end product demo.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from ..config import ModelConfig
from .blocks.espm import ESPM
from .blocks.pfem import PFEMSubmodule


class EPNet(nn.Module):
    """Efficient Pyramid Network for single-image super-resolution.

    Summary:
        Builds the two-branch EPNet backbone and the final reconstruction head.

    Paper mapping:
        Corresponds to the full EPNet architecture shown in the paper figure:
        shallow feature extraction -> PFEM pathway + ESPM pathway -> fusion ->
        convolution -> PixelShuffle output.

    Role in the system:
        This is the central model object consumed by training, evaluation,
        profiling, export, and inference.

    Implementation notes:
        The repository keeps the PFEM path sequential and the ESPM path parallel,
        then fuses both feature tensors by element-wise addition before the
        reconstruction head. This keeps the model lightweight and easy to deploy.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        # The shallow stem maps RGB pixels into the model feature space without
        # changing spatial resolution. This is the entry point for both branches.
        self.shallow = nn.Conv2d(
            self.config.in_channels,
            self.config.embed_dim,
            kernel_size=3,
            padding=1,
        )
        self.shared_pfem_block: PFEMSubmodule | None = None
        if self.config.share_pfem_weights:
            # Optional weight sharing is a repository-level efficiency variant.
            # It is not the main default, but it is useful for ablations and
            # edge-oriented experiments with tighter parameter budgets.
            self.shared_pfem_block = PFEMSubmodule(
                self.config.embed_dim,
                self.config.num_heads,
                self.config.window_size,
                self.config.mlp_ratio,
                global_context=self.config.global_context,
            )
            self.pfem_blocks = nn.ModuleList()
        else:
            # The default path mirrors the paper's repeated PFEM staging more
            # closely: each stage owns its own parameters.
            self.pfem_blocks = nn.ModuleList(
                PFEMSubmodule(
                    self.config.embed_dim,
                    self.config.num_heads,
                    self.config.window_size,
                    self.config.mlp_ratio,
                    global_context=self.config.global_context,
                )
                for _ in range(self.config.num_pfem)
            )
        self.espm = ESPM(
            self.config.embed_dim,
            self.config.espm_levels,
            self.config.split_ratio,
        )
        # Reconstruction follows the standard SR pattern: project to
        # C * upscale^2 channels and then rearrange them spatially with
        # PixelShuffle.
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
        """Return the feature tensors produced by the PFEM and ESPM branches.

        Args:
            x: Low-resolution input tensor with shape ``[B, C, H, W]``.

        Returns:
            A tuple ``(pfem_features, espm_features)`` where both tensors share
            the same shape ``[B, embed_dim, H, W]`` so they can be fused later.
        """
        base = self.shallow(x)
        pfem = base
        if self.shared_pfem_block is not None:
            # Shared-weight mode still applies the PFEM stage multiple times; it
            # only reuses parameters, not computation.
            for _ in range(self.config.num_pfem):
                pfem = self.shared_pfem_block(pfem)
        else:
            for block in self.pfem_blocks:
                pfem = block(pfem)
        # ESPM always branches from the shallow base features rather than from
        # the final PFEM output. This preserves a distinct multiscale context
        # pathway that is later fused with the detail pathway.
        espm = self.espm(base)
        return pfem, espm

    def forward(self, x: Tensor) -> Tensor:
        """Run the full EPNet forward pass.

        Shape notes:
            - input: ``[B, 3, H, W]``
            - output: ``[B, 3, H * upscale, W * upscale]``

        The PFEM and ESPM branches are fused by addition because both produce
        aligned feature tensors in the same embedding dimension.
        """
        pfem, espm = self.forward_features(x)
        return self.reconstruction(pfem + espm)

    def profile_input_shape(self) -> list[int]:
        """Return a representative input shape used for lightweight profiling."""
        return [1, self.config.in_channels, 48, 48]


def load_model_from_checkpoint(checkpoint: dict[str, object]) -> EPNet:
    """Rebuild an ``EPNet`` instance from a serialized checkpoint payload."""
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
    """Load a checkpoint dictionary with backward-compatible ``torch.load`` behavior.

    The helper prefers ``weights_only=True`` when the local PyTorch version
    supports it. That keeps deployment startup quieter and more future-proof.
    """
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_file}")
    try:
        checkpoint = torch.load(
            checkpoint_file,
            map_location=map_location,
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_file, map_location=map_location)

    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must deserialize to a dictionary.")
    return checkpoint
