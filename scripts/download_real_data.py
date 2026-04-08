from __future__ import annotations

import argparse
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path

DIV2K_URLS = {
    "DIV2K_train_HR.zip": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
    "DIV2K_valid_HR.zip": "https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip",
}

BENCHMARK_ZIP_URL = "https://cv.snu.ac.kr/research/VDSR/test_data.zip"
BENCHMARK_HF_ARCHIVES = {
    "Set5": ("eugenesiow/Set5", "data/Set5_HR.tar.gz"),
    "Set14": ("eugenesiow/Set14", "data/Set14_HR.tar.gz"),
    "BSD100": ("eugenesiow/BSD100", "data/BSD100_HR.tar.gz"),
    "Urban100": ("eugenesiow/Urban100", "data/Urban100_HR.tar.gz"),
}
MANGA109_NOTICE = (
    "Manga109 is not auto-downloaded by this script because it is distributed "
    "under a separate usage agreement. Place the extracted HR images under "
    "data/benchmarks/Manga109/HR manually if you have access."
)


def _is_valid_archive(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    suffixes = path.suffixes
    if suffixes and suffixes[-1] == ".zip":
        return zipfile.is_zipfile(path)
    if suffixes[-2:] == [".tar", ".gz"] or path.suffix == ".tgz":
        return tarfile.is_tarfile(path)
    return True


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and _is_valid_archive(destination):
        print(f"Skip existing file: {destination}")
        return
    if destination.exists():
        print(f"Removing invalid or partial archive: {destination}")
        destination.unlink()
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


def download_hf_benchmark_archives(downloads_root: Path, benchmarks_root: Path) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - exercised in real setup, not CI
        raise RuntimeError(
            "huggingface_hub is required for the benchmark fallback path. "
            "Install it with `pip install huggingface_hub`."
        ) from exc

    for benchmark_name, (repo_id, filename) in BENCHMARK_HF_ARCHIVES.items():
        archive_path = downloads_root / f"{benchmark_name}_HR.tar.gz"
        if not _is_valid_archive(archive_path):
            if archive_path.exists():
                archive_path.unlink()
            print(f"Downloading Hugging Face fallback for {benchmark_name}: {repo_id}/{filename}")
            source_path = Path(
                hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    repo_type="dataset",
                )
            )
            shutil.copy2(source_path, archive_path)
        extract_root = downloads_root / f"{benchmark_name}_hf_extract"
        shutil.rmtree(extract_root, ignore_errors=True)
        extract_root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive_path, "r:gz") as handle:
            handle.extractall(extract_root)
        _move_hr_images(extract_root, benchmarks_root / benchmark_name / "HR")


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

    try:
        benchmark_archive = downloads_root / "vdsr_test_data.zip"
        download(BENCHMARK_ZIP_URL, benchmark_archive)
        extracted_benchmarks = downloads_root / "vdsr_test_data"
        shutil.rmtree(extracted_benchmarks, ignore_errors=True)
        unzip(benchmark_archive, extracted_benchmarks)
        normalize_vdsr_benchmarks(extracted_benchmarks, benchmark_root)
    except Exception as exc:  # pragma: no cover - depends on external mirrors
        print(f"Official VDSR benchmark mirror failed: {exc}")
        print("Falling back to Hugging Face benchmark archives.")
        download_hf_benchmark_archives(downloads_root, benchmark_root)

    print(MANGA109_NOTICE)


if __name__ == "__main__":
    main()
