from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

DIV2K_URLS = {
    "DIV2K_train_HR.zip": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    "DIV2K_valid_HR.zip": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
}

BENCHMARK_ZIP_URL = "https://cv.snu.ac.kr/research/VDSR/test_data.zip"
MANGA109_NOTICE = (
    "Manga109 is not auto-downloaded by this script because it is distributed "
    "under a separate usage agreement. Place the extracted HR images under "
    "data/benchmarks/Manga109/HR manually if you have access."
)


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"Skip existing file: {destination}")
        return
    print(f"Downloading {url} -> {destination}")
    urllib.request.urlretrieve(url, destination)


def unzip(archive: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as zip_ref:
        zip_ref.extractall(target_dir)


def _move_hr_images(source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for image in source_dir.rglob("*"):
        if image.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
            shutil.copy2(image, target_dir / image.name)


def normalize_vdsr_benchmarks(extracted_root: Path, benchmarks_root: Path) -> None:
    alias_map = {
        "Set5": "Set5",
        "Set14": "Set14",
        "B100": "BSD100",
        "BSD100": "BSD100",
        "Urban100": "Urban100",
    }
    for source_name, normalized in alias_map.items():
        candidates = [path for path in extracted_root.rglob(source_name) if path.is_dir()]
        if not candidates:
            print(f"Manual benchmark fallback may be needed for {normalized}.")
            continue
        _move_hr_images(candidates[0], benchmarks_root / normalized / "HR")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download real EPNet training/eval datasets.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()

    raw_root = args.data_root / "raw"
    benchmark_root = args.data_root / "benchmarks"
    downloads_root = raw_root / "downloads"
    downloads_root.mkdir(parents=True, exist_ok=True)

    for filename, url in DIV2K_URLS.items():
        archive = downloads_root / filename
        download(url, archive)
        unzip(archive, raw_root / "DIV2K")

    benchmark_archive = downloads_root / "vdsr_test_data.zip"
    download(BENCHMARK_ZIP_URL, benchmark_archive)
    extracted_benchmarks = downloads_root / "vdsr_test_data"
    unzip(benchmark_archive, extracted_benchmarks)
    normalize_vdsr_benchmarks(extracted_benchmarks, benchmark_root)

    print(MANGA109_NOTICE)


if __name__ == "__main__":
    main()
