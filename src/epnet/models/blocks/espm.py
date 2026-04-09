"""ESPM implementation for EPNet's efficient multiscale context branch.

The paper describes ESPM as the branch that captures broader spatial structure
without drastically increasing computational cost. In this repository it is
implemented as:

- an entry DCAB block
- repeated pyramid downsampling stages
- attention-based top-down fusion back to the original resolution
- a final residual output projection

This file is one of the key places where the paper concept is translated into a
deployable engineering implementation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as torch_f
from torch import Tensor, nn

from .dcab import ChannelAttention, DynamicChannelAttentionBlock


class AttentionFusion(nn.Module):
    """Fuse same-scale and upsampled pyramid features with lightweight attention."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.attention = ChannelAttention(channels)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Fuse local-scale feature ``x`` with upsampled coarser context ``y``."""
        fused = self.fusion(torch.cat([x, y], dim=1))
        return fused * self.attention(fused) + x


class ESPM(nn.Module):
    """Efficient Spatial Pyramid Module.

    Paper mapping:
        Corresponds to the ESPM branch in the EPNet diagram.

    Role in the system:
        Provides multiscale context that complements the PFEM detail pathway.

    Implementation notes:
        The repository uses average pooling to build the pyramid and bilinear
        interpolation to return to finer resolutions. This is a clear and stable
        approximation of the paper's multilevel structure.
    """

    def __init__(self, channels: int, levels: int, split_ratio: float) -> None:
        super().__init__()
        if levels < 2:
            raise ValueError("levels must be >= 2")
        self.entry = DynamicChannelAttentionBlock(channels, split_ratio)
        self.pyramid_blocks = nn.ModuleList(
            DynamicChannelAttentionBlock(channels, split_ratio) for _ in range(levels - 1)
        )
        self.fusions = nn.ModuleList(AttentionFusion(channels) for _ in range(levels - 1))
        self.out_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        """Run the top-down ESPM pyramid and return an aligned residual feature map."""
        # Start the pyramid at the native shallow-feature resolution.
        pyramid: list[Tensor] = [self.entry(x)]
        features = pyramid[0]
        for block in self.pyramid_blocks:
            # Progressively build coarser representations. ``ceil_mode=True``
            # keeps odd input sizes safe during deployment and evaluation.
            features = torch_f.avg_pool2d(features, kernel_size=2, stride=2, ceil_mode=True)
            pyramid.append(block(features))

        # Fuse from coarse to fine so higher-level context is injected back into
        # the finer-resolution features before reconstruction.
        fused = pyramid[-1]
        for index in reversed(range(len(pyramid) - 1)):
            fused = torch_f.interpolate(
                fused,
                size=pyramid[index].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            fused = self.fusions[index](pyramid[index], fused)

        # The residual connection preserves the original shallow features and
        # makes the branch easier to optimize.
        return self.out_conv(fused) + x
