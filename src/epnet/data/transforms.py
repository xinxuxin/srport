"""Image loading and conversion helpers shared across training and deployment.

These helpers deliberately avoid heavy framework abstractions so the same pixel
conversion rules are easy to inspect from training, evaluation, and inference.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def list_images(directory: Path) -> list[Path]:
    """Recursively list supported image files under a directory."""
    return sorted(path for path in directory.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def load_image(path: Path) -> Image.Image:
    """Load an image as RGB."""
    return Image.open(path).convert("RGB")


def save_image(image: Image.Image, path: Path) -> None:
    """Save a PIL image, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert a PIL RGB image into a float tensor in ``[C, H, W]`` format."""
    array = np.asarray(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a normalized tensor back into a displayable RGB image."""
    clamped = tensor.detach().cpu().clamp(0.0, 1.0)
    array = (clamped.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)


def resize_bicubic(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Resize an image with bicubic interpolation."""
    return image.resize(size, resample=Image.Resampling.BICUBIC)


def make_divisible(value: int, divisor: int) -> int:
    """Round an integer upward so it is divisible by ``divisor``."""
    return int(math.ceil(value / divisor) * divisor)


def mod_crop(image: Image.Image, scale: int) -> Image.Image:
    """Crop an image so both spatial dimensions are divisible by ``scale``."""
    width = image.width - (image.width % scale)
    height = image.height - (image.height % scale)
    return image.crop((0, 0, width, height))
