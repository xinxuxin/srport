from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter
from torch import Tensor
from torch.utils.data import Dataset

from .utils import load_image, pil_to_tensor, resize_bicubic

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def list_images(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def synthetic_pattern_image(size: int, seed: int) -> Image.Image:
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


class TrainImageFolderDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        image_dir: Path,
        patch_size: int,
        scale: int,
        repeat: int = 16,
    ) -> None:
        self.images = list_images(image_dir)
        if not self.images:
            raise ValueError(f"No training images found in {image_dir}")
        self.patch_size = patch_size
        self.scale = scale
        self.repeat = repeat

    def __len__(self) -> int:
        return len(self.images) * self.repeat

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image = load_image(self.images[index % len(self.images)])
        hr_patch = self._random_crop(image)
        lr_size = (hr_patch.width // self.scale, hr_patch.height // self.scale)
        lr_patch = resize_bicubic(hr_patch, lr_size)
        return pil_to_tensor(lr_patch), pil_to_tensor(hr_patch)

    def _random_crop(self, image: Image.Image) -> Image.Image:
        crop_size = self.patch_size
        if image.width < crop_size or image.height < crop_size:
            new_width = max(crop_size, image.width)
            new_height = max(crop_size, image.height)
            image = resize_bicubic(image, (new_width, new_height))

        max_x = image.width - crop_size
        max_y = image.height - crop_size
        x = random.randint(0, max_x)
        y = random.randint(0, max_y)
        return image.crop((x, y, x + crop_size, y + crop_size))


class EvaluationImageDataset(Dataset[tuple[str, Tensor, Tensor]]):
    def __init__(self, hr_dir: Path, scale: int) -> None:
        self.images = list_images(hr_dir)
        if not self.images:
            raise ValueError(f"No evaluation images found in {hr_dir}")
        self.scale = scale

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[str, Tensor, Tensor]:
        path = self.images[index]
        hr = load_image(path)
        width = hr.width - (hr.width % self.scale)
        height = hr.height - (hr.height % self.scale)
        hr = hr.crop((0, 0, width, height))
        lr = resize_bicubic(hr, (width // self.scale, height // self.scale))
        return path.name, pil_to_tensor(lr), pil_to_tensor(hr)


class SyntheticPatternDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, count: int, hr_size: int, scale: int, seed: int = 42) -> None:
        self.count = count
        self.hr_size = hr_size
        self.scale = scale
        self.seed = seed

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image = synthetic_pattern_image(self.hr_size, self.seed + index)
        lr = resize_bicubic(image, (self.hr_size // self.scale, self.hr_size // self.scale))
        return pil_to_tensor(lr), pil_to_tensor(image)
