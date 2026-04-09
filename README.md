# EPNet Super-Resolution System

Production-oriented single-image super-resolution repository based on the paper
**"EPNet: An Efficient Pyramid Network for Enhanced Single-Image
Super-Resolution with Reduced Computational Requirements."**

This repository is intentionally broader than a paper reproduction. It includes:

- the EPNet model implementation
- real-data training and evaluation
- checkpoint export and profiling
- a FastAPI inference service
- a frontend demo application
- analytics, history, and deployment helpers

[English mirror for bilingual navigation](README.zh-CN.md)

## Project Overview

The repository solves one core problem: turning a trained EPNet
single-image-super-resolution model into a reproducible system that can be
trained, evaluated, exported, deployed, and presented clearly.

It is useful in three different contexts:

1. research and engineering comparison
2. local edge/deployment experimentation
3. presentation/demo workflows where a trained model must run end to end

## What the Repository Contains

- EPNet architecture with PFEM and ESPM branches
- model presets for paper-like and edge-oriented variants
- config-driven real-data training
- synthetic smoke/regression training
- benchmark evaluation with PSNR and SSIM
- checkpoint profiling and ONNX export smoke path
- deployment-oriented inference service with artifact selection
- Next.js frontend for upload, compare, analytics, and model explanation

## Repository Structure

```text
epnet/
├── configs/
│   ├── data/
│   ├── model/
│   └── train/
├── data/
│   ├── raw/
│   ├── processed/
│   └── benchmarks/
├── docs/
├── frontend/
├── outputs/
├── scripts/
├── src/
│   ├── epnet/
│   │   ├── data/
│   │   ├── models/
│   │   │   └── blocks/
│   │   └── utils/
│   └── epnet_api/
└── tests/
```

Key directories:

- `src/epnet/models`
  - core architecture and block implementations
- `src/epnet/data`
  - synthetic, DIV2K, benchmark, and paired dataset logic
- `src/epnet/train.py`
  - main training entrypoint
- `src/epnet/evaluate.py`
  - benchmark evaluation entrypoint
- `src/epnet/export.py`
  - ONNX export entrypoint
- `src/epnet_api`
  - FastAPI serving and runtime integration
- `frontend`
  - user-facing demo system
- `configs`
  - reproducible model/data/train settings
- `outputs`
  - checkpoints, logs, evaluation artifacts, profiles, and exports

## Architecture Overview

### Product Language

EPNet combines:

- a detail-focused enhancement branch
- a lightweight pyramid context branch
- a reconstruction head that converts fused features into a sharper output image

### Technical/Paper Language

The code follows the paper's high-level structure:

- shallow `3x3` convolution stem
- PFEM branch
- ESPM branch
- feature fusion
- reconstruction with convolution plus `PixelShuffle`

### Branch Responsibilities

PFEM:

- progressive feature refinement
- LFEB for local enhancement
- transformer-style context modeling
- ESAB for spatial attention refinement

ESPM:

- efficient pyramid-style multi-scale context
- DCAB-based split/fuse channel processing
- low-cost structural information pathway

Reconstruction head:

- fuse PFEM and ESPM outputs
- project into the upsampling tensor space
- `PixelShuffle` into x2/x3/x4 output resolution

## Paper-to-Code Mapping Summary

Top-level model:

- [/Users/macbook/Desktop/epnet/src/epnet/models/epnet.py](/Users/macbook/Desktop/epnet/src/epnet/models/epnet.py)

PFEM-related modules:

- [/Users/macbook/Desktop/epnet/src/epnet/models/blocks/pfem.py](/Users/macbook/Desktop/epnet/src/epnet/models/blocks/pfem.py)
- [/Users/macbook/Desktop/epnet/src/epnet/models/blocks/lfeb.py](/Users/macbook/Desktop/epnet/src/epnet/models/blocks/lfeb.py)
- [/Users/macbook/Desktop/epnet/src/epnet/models/blocks/global_context.py](/Users/macbook/Desktop/epnet/src/epnet/models/blocks/global_context.py)
- [/Users/macbook/Desktop/epnet/src/epnet/models/blocks/esab.py](/Users/macbook/Desktop/epnet/src/epnet/models/blocks/esab.py)

ESPM/DCAB:

- [/Users/macbook/Desktop/epnet/src/epnet/models/blocks/espm.py](/Users/macbook/Desktop/epnet/src/epnet/models/blocks/espm.py)
- [/Users/macbook/Desktop/epnet/src/epnet/models/blocks/dcab.py](/Users/macbook/Desktop/epnet/src/epnet/models/blocks/dcab.py)

Preset system:

- [/Users/macbook/Desktop/epnet/src/epnet/models/presets.py](/Users/macbook/Desktop/epnet/src/epnet/models/presets.py)
- [/Users/macbook/Desktop/epnet/src/epnet/models/registry.py](/Users/macbook/Desktop/epnet/src/epnet/models/registry.py)
- [/Users/macbook/Desktop/epnet/src/epnet/config.py](/Users/macbook/Desktop/epnet/src/epnet/config.py)

For the full mapping, read:

- [/Users/macbook/Desktop/epnet/docs/paper_to_code_map.md](/Users/macbook/Desktop/epnet/docs/paper_to_code_map.md)

## Presets

### `paper_like`

- closest to the paper-style structural baseline
- useful for comparison and explanation
- not the default deployment preset

### `edge_tiny`

- smallest preset
- good for smoke checks or very constrained deployment experiments

### `edge_default`

- repository default
- optimized for practical quality/efficiency balance
- current mainline x4 deployment artifact is trained from this preset

### `balanced_quality`

- wider comparison model
- useful when exploring local quality/latency trade-offs

## Training Workflow

### Synthetic Smoke Path

Use this when you want fast regression coverage, not meaningful benchmark
results.

```bash
PYTHONPATH=src .venv/bin/python -m epnet.train \
  --model-config configs/model/edge_default_x2.yaml \
  --data-config configs/data/synthetic_x2.yaml \
  --train-config configs/train/smoke.yaml
```

### Real-Data Mainline Path

1. Download datasets

```bash
PYTHONPATH=src .venv/bin/python scripts/download_real_data.py --data-root data
```

2. Prepare bicubic caches

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_real_data.py --data-root data --scales 2 3 4
```

3. Start or resume full x4 training

```bash
PYTHONPATH=src .venv/bin/python scripts/run_local_full_x4.py
```

4. Evaluate the trained artifact

```bash
PYTHONPATH=src .venv/bin/python -m epnet.evaluate \
  --checkpoint outputs/run_x4_edge_default/inference.pt \
  --data-config configs/data/div2k_x4.yaml \
  --output-json outputs/run_x4_edge_default/eval.json \
  --output-markdown outputs/run_x4_edge_default/eval.md
```

### Resume Behavior

The full training scripts automatically resume from `latest.pt` when present.
This applies to the mainline x4 run under:

- `/Users/macbook/Desktop/epnet/outputs/run_x4_edge_default`

### Training Outputs

Typical training outputs:

- `manifest.json`
- `train_log.jsonl`
- `latest.pt`
- `best.pt`
- `inference.pt`
- `stepXXXX.pt`
- `eval.json`
- `eval.md`
- `profile.json`
- `profile.md`
- `model.onnx`

## Evaluation Workflow

Evaluation uses:

- PSNR
- SSIM

Supported benchmark datasets:

- Set5
- Set14
- BSD100
- Urban100
- Manga109 if manually provided

Main evaluation entrypoint:

- [/Users/macbook/Desktop/epnet/src/epnet/evaluate.py](/Users/macbook/Desktop/epnet/src/epnet/evaluate.py)

## Deployment Workflow

### Default Deployment Artifact

The backend prefers:

- `/Users/macbook/Desktop/epnet/outputs/run_x4_edge_default/inference.pt`

If that file is missing, it falls back to:

- `/Users/macbook/Desktop/epnet/checkpoints/demo_x4.pt`

### Runtime Backend

Default backend:

- PyTorch

Optional backend:

- ONNX, when export exists and the runtime environment supports it

### One-Command Demo Path

```bash
./scripts/run_local.sh
```

If default ports are already occupied:

```bash
EPNET_API_PORT=8012 EPNET_FRONTEND_PORT=3012 ./scripts/run_local.sh
```

### Useful Endpoints

- frontend: `http://localhost:3012`
- model page: `http://localhost:3012/model`
- API health: `http://localhost:8012/api/v1/health`
- model info: `http://localhost:8012/api/v1/model/info`

### Example Inference Request

```bash
curl -X POST http://localhost:8012/api/v1/infer \
  -F 'file=@data/samples/demo_input.png;type=image/png' \
  -F 'session_id=demo' \
  -F 'method=epnet' \
  -F 'scale=4' \
  -F 'output_format=PNG' \
  -F 'tile_size=0'
```

The output image URL is returned in the JSON response.

## Start Here If You Want to Understand the Code

### Model

Start with:

1. [/Users/macbook/Desktop/epnet/src/epnet/models/epnet.py](/Users/macbook/Desktop/epnet/src/epnet/models/epnet.py)
2. [/Users/macbook/Desktop/epnet/src/epnet/models/blocks/pfem.py](/Users/macbook/Desktop/epnet/src/epnet/models/blocks/pfem.py)
3. [/Users/macbook/Desktop/epnet/src/epnet/models/blocks/espm.py](/Users/macbook/Desktop/epnet/src/epnet/models/blocks/espm.py)

### Training

Start with:

1. [/Users/macbook/Desktop/epnet/src/epnet/train.py](/Users/macbook/Desktop/epnet/src/epnet/train.py)
2. [/Users/macbook/Desktop/epnet/src/epnet/data/datamodule.py](/Users/macbook/Desktop/epnet/src/epnet/data/datamodule.py)
3. [/Users/macbook/Desktop/epnet/src/epnet/data/paired_dataset.py](/Users/macbook/Desktop/epnet/src/epnet/data/paired_dataset.py)

### Deployment

Start with:

1. [/Users/macbook/Desktop/epnet/src/epnet_api/app/services/inference.py](/Users/macbook/Desktop/epnet/src/epnet_api/app/services/inference.py)
2. [/Users/macbook/Desktop/epnet/src/epnet_api/app/api/routes.py](/Users/macbook/Desktop/epnet/src/epnet_api/app/api/routes.py)
3. [/Users/macbook/Desktop/epnet/src/epnet_api/app/core/settings.py](/Users/macbook/Desktop/epnet/src/epnet_api/app/core/settings.py)

## Config and Dataset Guides

- config reference:
  [/Users/macbook/Desktop/epnet/docs/config_reference.md](/Users/macbook/Desktop/epnet/docs/config_reference.md)
- dataset setup:
  [/Users/macbook/Desktop/epnet/docs/dataset_setup.md](/Users/macbook/Desktop/epnet/docs/dataset_setup.md)
- real training workflow:
  [/Users/macbook/Desktop/epnet/docs/real_training_workflow.md](/Users/macbook/Desktop/epnet/docs/real_training_workflow.md)
- code reading guide:
  [/Users/macbook/Desktop/epnet/docs/code_reading_guide.md](/Users/macbook/Desktop/epnet/docs/code_reading_guide.md)
- presentation notes:
  [/Users/macbook/Desktop/epnet/docs/design_notes_for_presentation.md](/Users/macbook/Desktop/epnet/docs/design_notes_for_presentation.md)

## Known Limitations

- The repository is a documented, engineering-grounded EPNet implementation,
  not an official author release.
- Some paper details remain approximations; see
  [/Users/macbook/Desktop/epnet/docs/epnet_assumptions.md](/Users/macbook/Desktop/epnet/docs/epnet_assumptions.md).
- ONNX export is supported as a smoke/deployment-preparation path, but runtime
  support depends on the local ONNX stack.
- Manga109 is supported only through manual dataset placement.
- Synthetic experiments are useful for engineering comparison, not benchmark
  claims.

## Optimization Opportunities

- validate ONNXRuntime on more deployment targets
- add stricter runtime benchmarking across CPU, MPS, and CUDA
- investigate quantization-friendly alternatives for the global-context block
- add downstream OCR or inspection tasks to show task-level value after SR
- profile tile inference on larger deployment-style inputs

## Related Docs

- assumptions:
  [/Users/macbook/Desktop/epnet/docs/epnet_assumptions.md](/Users/macbook/Desktop/epnet/docs/epnet_assumptions.md)
- model audit:
  [/Users/macbook/Desktop/epnet/docs/model_audit_report.md](/Users/macbook/Desktop/epnet/docs/model_audit_report.md)
- deployment report:
  [/Users/macbook/Desktop/epnet/docs/system_deployment_report.md](/Users/macbook/Desktop/epnet/docs/system_deployment_report.md)
- first real x4 run report:
  [/Users/macbook/Desktop/epnet/docs/first_real_x4_run_report.md](/Users/macbook/Desktop/epnet/docs/first_real_x4_run_report.md)
