# Dataset Setup

## Directory Convention

```text
data/
  raw/
    DIV2K/
      DIV2K_train_HR/
      DIV2K_valid_HR/
  processed/
  benchmarks/
    Set5/HR/
    Set14/HR/
    BSD100/HR/
    Urban100/HR/
    Manga109/HR/
```

## Automatic Downloads

The repository includes:

- `scripts/download_real_data.py`
- `scripts/prepare_real_data.py`

Supported automatic paths:

- DIV2K train and validation HR zips
- VDSR benchmark test pack, used to normalize:
  - Set5
  - Set14
  - BSD100
  - Urban100

## Manual Fallback

Manga109 is not auto-downloaded by default in this repository because it is distributed under its own usage terms.

If you have access:

1. Download Manga109 manually from the official distribution site.
2. Extract the HR images.
3. Place them in:
   - `data/benchmarks/Manga109/HR/`

## Preparation Step

To generate cached bicubic LR images:

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_real_data.py --data-root data --scales 2 3 4
```

This creates reusable LR caches under `data/processed/`.

## Validation Behavior

The training scripts fail clearly if required real-data paths are missing. They do not silently skip missing datasets.
