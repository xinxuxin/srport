"""Enhanced spatial attention block used at the end of each PFEM stage."""

from __future__ import annotations

import torch.nn.functional as torch_f
from torch import Tensor, nn


class ESAB(nn.Module):
    """Enhanced Spatial Attention Block.

    Paper mapping:
        This is the PFEM submodule's final spatial reweighting component.

    Implementation notes:
        The block uses pooled context and a compact bottleneck to keep the
        attention path cheaper than a full-resolution heavy branch.
    """

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
        """Compute spatial attention from pooled context and apply it residually."""
        residual = x
        features = self.head(x)
        # Pool before the refinement stack to keep the attention branch compact.
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
