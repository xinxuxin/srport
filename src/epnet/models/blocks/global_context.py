from __future__ import annotations

import torch
import torch.nn.functional as torch_f
from torch import Tensor, nn


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
        index = self.relative_position_index.reshape(-1).to(torch.long)  # type: ignore[operator]
        relative_position_bias = self.relative_position_bias_table[index]
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


class LightweightConvContext(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.context = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.context(x)


class IdentityContext(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


def build_global_context(
    kind: str,
    channels: int,
    num_heads: int,
    window_size: int,
    mlp_ratio: float,
) -> nn.Module:
    if kind == "swin":
        return ModifiedSwinTransformer(channels, num_heads, window_size, mlp_ratio)
    if kind == "lightweight_conv_context":
        return LightweightConvContext(channels)
    if kind == "none":
        return IdentityContext()
    raise ValueError(f"Unsupported global context kind: {kind}")
