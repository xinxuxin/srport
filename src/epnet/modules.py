from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as torch_f
from torch import Tensor, nn


class ECAM(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            1,
            1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False,
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        pooled = self.avg_pool(x).squeeze(-1).transpose(1, 2)
        weights = self.conv(pooled).transpose(1, 2).unsqueeze(-1)
        return x * self.sigmoid(weights)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        return self.sigmoid(self.proj(self.pool(x)))


class LFEB(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.ecam = ECAM(channels)

    def forward(self, x: Tensor) -> Tensor:
        features = self.conv2(self.act(self.conv1(x)))
        features = self.ecam(features)
        return x + features


def window_partition(x: Tensor, window_size: int) -> Tensor:
    batch, height, width, channels = x.shape
    x = x.view(
        batch,
        height // window_size,
        window_size,
        width // window_size,
        window_size,
        channels,
    )
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size * window_size, channels)


def window_reverse(windows: Tensor, window_size: int, height: int, width: int) -> Tensor:
    batch = int(windows.shape[0] / ((height // window_size) * (width // window_size)))
    x = windows.view(
        batch,
        height // window_size,
        width // window_size,
        window_size,
        window_size,
        -1,
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(batch, height, width, -1)


class Mlp(nn.Module):
    def __init__(self, channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(channels, hidden_channels)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_channels, channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.act(self.fc1(x)))


class WindowAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int, window_size: int) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")
        self.channels = channels
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (channels // num_heads) ** -0.5
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)

        relative_size = (2 * window_size - 1) * (2 * window_size - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(relative_size, num_heads))

        coords = torch.arange(window_size)
        coords_flatten = torch.stack(torch.meshgrid(coords, coords, indexing="ij")).flatten(1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, x: Tensor) -> Tensor:
        batch_windows, tokens, channels = x.shape
        qkv = (
            self.qkv(x)
            .reshape(batch_windows, tokens, 3, self.num_heads, channels // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv[0], qkv[1], qkv[2]
        query = query * self.scale

        attention = query @ key.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ]
        relative_position_bias = relative_position_bias.view(tokens, tokens, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).unsqueeze(0)
        attention = attention + relative_position_bias
        attention = attention.softmax(dim=-1)

        output = (attention @ value).transpose(1, 2).reshape(batch_windows, tokens, channels)
        return self.proj(output)


class SwinBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        window_size: int,
        shift_size: int,
        mlp_ratio: float,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.norm1 = nn.LayerNorm(channels)
        self.attn = WindowAttention(channels, num_heads, window_size)
        self.norm2 = nn.LayerNorm(channels)
        hidden_channels = int(channels * mlp_ratio)
        self.mlp = Mlp(channels, hidden_channels)

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        pad_h = (self.window_size - height % self.window_size) % self.window_size
        pad_w = (self.window_size - width % self.window_size) % self.window_size
        if pad_h or pad_w:
            x = torch_f.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        _, _, padded_h, padded_w = x.shape
        features = x.permute(0, 2, 3, 1).contiguous()
        residual = features

        if self.shift_size:
            shifted = torch.roll(features, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted = features

        normalized = self.norm1(shifted)
        windows = window_partition(normalized, self.window_size)
        attended = self.attn(windows)
        shifted_back = window_reverse(attended, self.window_size, padded_h, padded_w)

        if self.shift_size:
            shifted_back = torch.roll(
                shifted_back,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),
            )

        features = residual + shifted_back
        features = features + self.mlp(self.norm2(features))
        features = features.permute(0, 3, 1, 2).contiguous()

        if pad_h or pad_w:
            features = features[:, :, :height, :width]
        return features


class ModifiedSwinTransformer(nn.Module):
    def __init__(self, channels: int, num_heads: int, window_size: int, mlp_ratio: float) -> None:
        super().__init__()
        self.blocks = nn.Sequential(
            SwinBlock(channels, num_heads, window_size, 0, mlp_ratio),
            SwinBlock(channels, num_heads, window_size, window_size // 2, mlp_ratio),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


class ESAB(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        bottleneck = max(8, channels // 4)
        self.head = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.refine = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, bottleneck, kernel_size=1),
            nn.Conv2d(bottleneck, bottleneck, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Conv2d(bottleneck, channels, kernel_size=1)
        self.gate = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        features = self.head(x)
        pooled = torch_f.max_pool2d(features, kernel_size=7, stride=3, padding=3)
        refined = self.refine(pooled)
        upsampled = torch_f.interpolate(
            refined,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        attention = self.gate(self.project(upsampled))
        return residual + features * attention


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
        pyramid: List[Tensor] = [self.entry(x)]
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


class PFEMSubmodule(nn.Module):
    def __init__(self, channels: int, num_heads: int, window_size: int, mlp_ratio: float) -> None:
        super().__init__()
        self.lfeb = LFEB(channels)
        self.transformer = ModifiedSwinTransformer(channels, num_heads, window_size, mlp_ratio)
        self.esab = ESAB(channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.lfeb(x)
        x = self.transformer(x)
        return self.esab(x)


@dataclass(frozen=True)
class FeatureMapShape:
    batch: int
    channels: int
    height: int
    width: int
