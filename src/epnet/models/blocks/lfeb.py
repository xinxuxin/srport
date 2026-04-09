"""Local feature extraction block used at the start of each PFEM stage."""

from __future__ import annotations

from torch import Tensor, nn


class ECAM(nn.Module):
    """Efficient channel attention module used inside LFEB."""

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
        """Reweight channels after lightweight global pooling."""
        pooled = self.avg_pool(x).squeeze(-1).transpose(1, 2)
        weights = self.conv(pooled).transpose(1, 2).unsqueeze(-1)
        return x * self.sigmoid(weights)


class LFEB(nn.Module):
    """Local Feature Extraction Block.

    Paper mapping:
        Corresponds to the convolutional front of the PFEM submodule.

    Role in the system:
        Strengthens local textures before the global-context block is applied.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.ecam = ECAM(channels)

    def forward(self, x: Tensor) -> Tensor:
        """Apply local convolutions, then channel attention, then add a residual."""
        features = self.conv2(self.act(self.conv1(x)))
        features = self.ecam(features)
        return x + features
