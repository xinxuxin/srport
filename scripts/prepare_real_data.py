# ruff: noqa: E402
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from epnet.data import list_images, load_image, mod_crop, resize_bicubic


def prepare_cache(hr_root: Path, processed_root: Path, scales: list[int]) -> None:
    images = list_images(hr_root)
    if not images:
        raise FileNotFoundError(f"No HR images found in {hr_root}")
    for image_path in images:
        hr = load_image(image_path)
        for scale in scales:
            cropped = mod_crop(hr, scale)
            lr = resize_bicubic(cropped, (cropped.width // scale, cropped.height // scale))
            relative = image_path.relative_to(hr_root)
            output_path = processed_root / f"X{scale}" / relative
            output_path.parent.mkdir(parents=True, exist_ok=True)
            lr.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare bicubic LR caches for real datasets.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--scales", type=int, nargs="+", default=[2, 3, 4])
    args = parser.parse_args()

    div2k_root = args.data_root / "raw" / "DIV2K"
    prepare_cache(
        div2k_root / "DIV2K_train_HR",
        args.data_root / "processed" / "DIV2K_train_LR_bicubic",
        args.scales,
    )
    prepare_cache(
        div2k_root / "DIV2K_valid_HR",
        args.data_root / "processed" / "DIV2K_valid_LR_bicubic",
        args.scales,
    )
    for benchmark_name in ("Set5", "Set14", "BSD100", "Urban100", "Manga109"):
        hr_dir = args.data_root / "benchmarks" / benchmark_name / "HR"
        if hr_dir.exists():
            prepare_cache(
                hr_dir,
                args.data_root / "processed" / "benchmarks" / benchmark_name / "LR_bicubic",
                args.scales,
            )
        else:
            print(f"Skip missing benchmark HR directory: {hr_dir}")


if __name__ == "__main__":
    main()
