"""Training and evaluation datasets that pair LR and HR images.

The repository trains from HR images and synthesizes LR inputs on the fly using
bicubic downsampling. That keeps the training path reproducible and makes the LR
generation strategy explicit in code instead of relying on hidden preprocessed
artifacts.
"""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from .transforms import list_images, load_image, pil_to_tensor, resize_bicubic


class TrainImageFolderDataset(Dataset[tuple[Tensor, Tensor]]):
    """Patch-based training dataset backed by a folder of HR images."""

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
        """Return a bicubic LR patch and its HR supervision target."""
        image = load_image(self.images[index % len(self.images)])
        hr_patch = self._random_crop(image)
        lr_size = (hr_patch.width // self.scale, hr_patch.height // self.scale)
        lr_patch = resize_bicubic(hr_patch, lr_size)
        return pil_to_tensor(lr_patch), pil_to_tensor(hr_patch)

    def _random_crop(self, image: Image.Image) -> Image.Image:
        """Sample a square HR crop, resizing tiny images upward if necessary."""
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


class PairedSuperResolutionDataset(Dataset[tuple[str, Tensor, Tensor]]):
    """Evaluation dataset that returns image name, LR tensor, and HR tensor."""

    def __init__(self, hr_paths: list[Path], scale: int) -> None:
        if not hr_paths:
            raise ValueError("PairedSuperResolutionDataset requires at least one HR image.")
        self.hr_paths = sorted(hr_paths)
        self.scale = scale

    def __len__(self) -> int:
        return len(self.hr_paths)

    def __getitem__(self, index: int) -> tuple[str, Tensor, Tensor]:
        """Apply mod-crop and bicubic downsampling to form a deterministic pair."""
        path = self.hr_paths[index]
        hr = load_image(path)
        width = hr.width - (hr.width % self.scale)
        height = hr.height - (hr.height % self.scale)
        hr = hr.crop((0, 0, width, height))
        lr = resize_bicubic(hr, (width // self.scale, height // self.scale))
        return path.name, pil_to_tensor(lr), pil_to_tensor(hr)
