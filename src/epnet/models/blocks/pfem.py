"""PFEM implementation used by the EPNet detail branch.

PFEM stands for Panoramic Feature Extraction Module in the paper. In this
repository each PFEM stage is modeled as a compact three-part sequence:

1. LFEB for local convolutional feature extraction
2. a pluggable global-context block (default: modified Swin-style attention)
3. ESAB for spatial attention refinement

This keeps the code close to the paper's narrative while remaining friendly to
ablation studies and deployment-oriented variants.
"""

from __future__ import annotations

from torch import Tensor, nn

from .esab import ESAB
from .global_context import build_global_context
from .lfeb import LFEB


class PFEMSubmodule(nn.Module):
    """One repeatable PFEM stage.

    Paper mapping:
        This block corresponds to the PFEM submodule inset shown in the EPNet
        paper figure.

    Shape notes:
        Input and output both use ``[B, C, H, W]``. The module preserves spatial
        size and channel count so multiple PFEM stages can be stacked directly.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        global_context: str = "swin",
    ) -> None:
        super().__init__()
        # LFEB focuses on local texture recovery with convolutional processing.
        self.lfeb = LFEB(channels)
        # The global-context block is pluggable so the repository can compare
        # the paper-like Swin-inspired path against lighter deployment variants.
        self.transformer = build_global_context(
            global_context,
            channels,
            num_heads,
            window_size,
            mlp_ratio,
        )
        # ESAB reweights spatial locations after local/global context mixing.
        self.esab = ESAB(channels)

    def forward(self, x: Tensor) -> Tensor:
        """Process one PFEM stage in the same order used in the paper figure."""
        x = self.lfeb(x)
        x = self.transformer(x)
        return self.esab(x)
