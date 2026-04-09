"""Dynamic channel attention utilities used inside ESPM.

DCAB is one of the paper's characteristic components inside the ESPM branch. In
this repository it is implemented as a practical split-attend-recombine block:

- project channels
- split them into two groups
- apply channel attention to each side independently
- create crossed combinations
- fuse the result back with a residual connection
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ChannelAttention(nn.Module):
    """Simple channel attention gate using global average pooling."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        """Return per-channel weights with shape ``[B, C, 1, 1]``."""
        return self.sigmoid(self.proj(self.pool(x)))


class DynamicChannelAttentionBlock(nn.Module):
    """Deployment-friendly DCAB approximation used by ESPM.

    Shape notes:
        Input and output use ``[B, C, H, W]`` and preserve both spatial and
        channel dimensions.
    """

    def __init__(self, channels: int, split_ratio: float = 0.5) -> None:
        super().__init__()
        split_channels = max(1, min(channels - 1, int(channels * split_ratio)))
        self.left_channels = split_channels
        self.right_channels = channels - split_channels
        self.pre = nn.Conv2d(channels, channels, kernel_size=1)
        self.left_attn = ChannelAttention(self.left_channels)
        self.right_attn = ChannelAttention(self.right_channels)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Split channels, attend to both sides, cross-fuse, and add a residual."""
        residual = x
        projected = self.pre(x)
        # The split ratio controls how much capacity each branch receives. This
        # is one of the simplest knobs for balancing accuracy and efficiency.
        left, right = torch.split(projected, [self.left_channels, self.right_channels], dim=1)
        alpha = self.left_attn(left)
        beta = self.right_attn(right)
        # Cross-over concatenations approximate the paper figure's split/rejoin
        # communication pattern without introducing complicated routing logic.
        crossover_a = torch.cat([alpha * left, right], dim=1)
        crossover_b = torch.cat([left, beta * right], dim=1)
        fused = self.fuse(torch.cat([crossover_a, crossover_b], dim=1))
        return residual + fused
