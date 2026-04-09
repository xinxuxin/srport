"""Synthetic smoke-data utilities for fast regression testing.

Synthetic data is no longer the main research path in this repository, but it
remains valuable for CI-friendly tests and quick local sanity checks. The images
here are deliberately structured enough to exercise super-resolution behavior
without requiring real datasets.
"""

from __future__ import annotations

import random

from PIL import Image, ImageDraw, ImageFilter
from torch import Tensor
from torch.utils.data import Dataset

from .transforms import pil_to_tensor, resize_bicubic


def synthetic_pattern_image(size: int, seed: int) -> Image.Image:
    """Create a deterministic synthetic image with edges, lines, and shapes."""
    generator = random.Random(seed)
    image = Image.new(
        "RGB",
        (size, size),
        color=tuple(generator.randint(0, 255) for _ in range(3)),
    )
    draw = ImageDraw.Draw(image)
    for _ in range(16):
        color = tuple(generator.randint(0, 255) for _ in range(3))
        x0 = generator.randint(0, size - 8)
        y0 = generator.randint(0, size - 8)
        x1 = generator.randint(x0 + 4, size)
        y1 = generator.randint(y0 + 4, size)
        if generator.random() < 0.5:
            draw.rectangle((x0, y0, x1, y1), outline=color, width=2)
        else:
            draw.ellipse((x0, y0, x1, y1), outline=color, width=2)

    for _ in range(12):
        x0 = generator.randint(0, size - 1)
        y0 = generator.randint(0, size - 1)
        x1 = generator.randint(0, size - 1)
        y1 = generator.randint(0, size - 1)
        color = tuple(generator.randint(0, 255) for _ in range(3))
        draw.line((x0, y0, x1, y1), fill=color, width=generator.randint(1, 3))

    return image.filter(ImageFilter.SHARPEN)


class SyntheticPatternDataset(Dataset[tuple[Tensor, Tensor]]):
    """Small synthetic SR dataset used by smoke tests and fast experiments."""

    def __init__(self, count: int, hr_size: int, scale: int, seed: int = 42) -> None:
        self.count = count
        self.hr_size = hr_size
        self.scale = scale
        self.seed = seed

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """Return a deterministic synthetic LR/HR pair."""
        image = synthetic_pattern_image(self.hr_size, self.seed + index)
        lr = resize_bicubic(image, (self.hr_size // self.scale, self.hr_size // self.scale))
        return pil_to_tensor(lr), pil_to_tensor(image)
