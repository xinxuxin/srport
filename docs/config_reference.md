# Config Reference

This repository uses three config families:

- model configs
- data configs
- train configs

Each config is intentionally small so the main logic stays in code while the
experiment identity stays reproducible in YAML.

## How Configs Work Together

One training run usually combines:

1. a model config from `configs/model`
2. a data config from `configs/data`
3. a training schedule config from `configs/train`

Example:

```bash
PYTHONPATH=src .venv/bin/python -m epnet.train \
  --model-config configs/model/edge_default.yaml \
  --data-config configs/data/div2k_x4.yaml \
  --train-config configs/train/local_full_x4.yaml
```

## Model Configs

Location:

- `/Users/macbook/Desktop/epnet/configs/model`

Files:

- `paper_like.yaml`
- `edge_tiny.yaml`
- `edge_default.yaml`
- `balanced_quality.yaml`
- `edge_default_x2.yaml`

Key fields:

- `preset`
  - selects the preset family defined in
    [/Users/macbook/Desktop/epnet/src/epnet/models/presets.py](/Users/macbook/Desktop/epnet/src/epnet/models/presets.py)
- `upscale`
  - sets x2, x3, or x4

When to use each preset:

- `paper_like`
  - baseline comparison
  - best when you want to discuss paper similarity
- `edge_tiny`
  - smallest variant
  - useful for smoke tests and highly constrained deployment
- `edge_default`
  - recommended default
  - best balance for the current product/demo path
- `balanced_quality`
  - larger local comparison point when quality is prioritized

## Data Configs

Location:

- `/Users/macbook/Desktop/epnet/configs/data`

Files:

- `synthetic_x2.yaml`
- `div2k_x2.yaml`
- `div2k_x4.yaml`

Key fields:

- `dataset_type`
  - `synthetic` for smoke tests
  - `div2k` for the real-data mainline
- `train_hr_dir`
  - location of high-resolution training images
- `processed_root`
  - cache directory for generated bicubic LR images
- `scale`
  - target upscale factor
- `benchmark_roots`
  - benchmark datasets for evaluation

Practical guidance:

- Use `synthetic_x2.yaml` only for smoke/regression checks.
- Use `div2k_x4.yaml` as the mainline real-data path.
- Use `div2k_x2.yaml` for faster local experiments or scale-specific comparisons.

## Train Configs

Location:

- `/Users/macbook/Desktop/epnet/configs/train`

Files:

- `smoke.yaml`
- `local_full_x2.yaml`
- `local_full_x4.yaml`

Key fields:

- `scale`
  - must match the chosen model/data config
- `patch_size`
  - HR crop size seen by the training loss
- `batch_size`
  - local-memory-sensitive training control
- `total_steps`
  - full training horizon
- `save_every`
  - checkpoint cadence
- `val_every`
  - validation cadence
- `device`
  - `auto`, `cpu`, `cuda`, or `mps`
- `amp`
  - AMP policy
- `run_name`
  - output directory name under `outputs/`
- `auto_resume`
  - automatically continue from `latest.pt` when present

How to interpret these configs:

- `smoke.yaml`
  - synthetic-only validation of the pipeline
  - should stay fast and CI-friendly
- `local_full_x2.yaml`
  - real-data local run at x2
- `local_full_x4.yaml`
  - primary full training schedule for the mainline product artifact

## Smoke vs Full Training

Smoke path:

- small synthetic datasets
- very short schedules
- used to validate that code still works

Full path:

- real DIV2K data
- long resumable schedule
- produces the deployment checkpoint used by the API/frontend system

## x2 vs x4

x2:

- faster training and inference
- useful for debugging or controlled comparisons

x4:

- main presentation target for this repository
- harder reconstruction problem
- default deployment artifact comes from the x4 mainline run

## Deployment Artifact Selection

Deployment configuration is not primarily stored in YAML. It is handled through
backend runtime settings:

- [/Users/macbook/Desktop/epnet/src/epnet_api/app/core/settings.py](/Users/macbook/Desktop/epnet/src/epnet_api/app/core/settings.py)

Important runtime environment variables:

- `EPNET_CHECKPOINT_PATH`
- `EPNET_CHECKPOINT_DIR`
- `EPNET_INFERENCE_BACKEND`
- `EPNET_ONNX_MODEL_PATH`
- `EPNET_ARTIFACTS_DIR`

The backend prefers the trained x4 inference checkpoint under:

- `/Users/macbook/Desktop/epnet/outputs/run_x4_edge_default/inference.pt`

## Common Config Combinations

Smoke regression:

```bash
PYTHONPATH=src .venv/bin/python -m epnet.train \
  --model-config configs/model/edge_default_x2.yaml \
  --data-config configs/data/synthetic_x2.yaml \
  --train-config configs/train/smoke.yaml
```

Mainline real-data x4:

```bash
PYTHONPATH=src .venv/bin/python -m epnet.train \
  --model-config configs/model/edge_default.yaml \
  --data-config configs/data/div2k_x4.yaml \
  --train-config configs/train/local_full_x4.yaml
```

Mainline evaluation:

```bash
PYTHONPATH=src .venv/bin/python -m epnet.evaluate \
  --checkpoint outputs/run_x4_edge_default/inference.pt \
  --data-config configs/data/div2k_x4.yaml \
  --output-json outputs/run_x4_edge_default/eval.json \
  --output-markdown outputs/run_x4_edge_default/eval.md
```
