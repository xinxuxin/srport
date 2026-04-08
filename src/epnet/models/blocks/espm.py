from __future__ import annotations

import torch
import torch.nn.functional as torch_f
from torch import Tensor, nn

from .dcab import ChannelAttention, DynamicChannelAttentionBlock


class AttentionFusion(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.attention = ChannelAttention(channels)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        fused = self.fusion(torch.cat([x, y], dim=1))
        return fused * self.attention(fused) + x


class ESPM(nn.Module):
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
        pyramid: list[Tensor] = [self.entry(x)]
        features = pyramid[0]
        for block in self.pyramid_blocks:
            features = torch_f.avg_pool2d(features, kernel_size=2, stride=2, ceil_mode=True)
            pyramid.append(block(features))

        fused = pyramid[-1]
        for index in reversed(range(len(pyramid) - 1)):
            fused = torch_f.interpolate(
                fused,
                size=pyramid[index].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            fused = self.fusions[index](pyramid[index], fused)

        return self.out_conv(fused) + x
