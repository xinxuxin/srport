from __future__ import annotations

import torch
from torch import Tensor, nn


class ChannelAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        return self.sigmoid(self.proj(self.pool(x)))


class DynamicChannelAttentionBlock(nn.Module):
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
        residual = x
        projected = self.pre(x)
        left, right = torch.split(projected, [self.left_channels, self.right_channels], dim=1)
        alpha = self.left_attn(left)
        beta = self.right_attn(right)
        crossover_a = torch.cat([alpha * left, right], dim=1)
        crossover_b = torch.cat([left, beta * right], dim=1)
        fused = self.fuse(torch.cat([crossover_a, crossover_b], dim=1))
        return residual + fused
