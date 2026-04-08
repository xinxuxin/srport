from __future__ import annotations

from torch import Tensor, nn

from .esab import ESAB
from .global_context import build_global_context
from .lfeb import LFEB


class PFEMSubmodule(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        global_context: str = "swin",
    ) -> None:
        super().__init__()
        self.lfeb = LFEB(channels)
        self.transformer = build_global_context(
            global_context,
            channels,
            num_heads,
            window_size,
            mlp_ratio,
        )
        self.esab = ESAB(channels)

    def forward(self, x: Tensor) -> Tensor:
        x = self.lfeb(x)
        x = self.transformer(x)
        return self.esab(x)
