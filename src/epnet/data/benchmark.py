from __future__ import annotations

from pathlib import Path

from torch import Tensor
from torch.utils.data import Dataset

from .paired_dataset import PairedSuperResolutionDataset
from .transforms import list_images


class EvaluationImageDataset(Dataset[tuple[str, Tensor, Tensor]]):
    def __init__(self, hr_dir: Path, scale: int) -> None:
        self.images = list_images(hr_dir)
        if not self.images:
            raise ValueError(f"No evaluation images found in {hr_dir}")
        self.scale = scale
        self.dataset = PairedSuperResolutionDataset(self.images, scale)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[str, Tensor, Tensor]:
        return self.dataset[index]
