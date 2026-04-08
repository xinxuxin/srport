from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = deepcopy(model).eval()
        for parameter in self.shadow.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        shadow_params = dict(self.shadow.named_parameters())
        for name, parameter in model.named_parameters():
            shadow_params[name].mul_(self.decay).add_(parameter.data, alpha=1.0 - self.decay)

        shadow_buffers = dict(self.shadow.named_buffers())
        for name, buffer in model.named_buffers():
            shadow_buffers[name].copy_(buffer)
