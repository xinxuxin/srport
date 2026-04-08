from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def save_image(image: Image.Image, path: Path) -> None:
    ensure_dir(path.parent)
    image.save(path)


def pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    clamped = tensor.detach().cpu().clamp(0.0, 1.0)
    array = (clamped.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)


def resize_bicubic(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    return image.resize(size, resample=Image.Resampling.BICUBIC)


def make_divisible(value: int, divisor: int) -> int:
    return int(math.ceil(value / divisor) * divisor)
