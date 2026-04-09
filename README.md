# EPNet Demo

Production-oriented single-image super-resolution demo based on the paper **"EPNet: An Efficient Pyramid Network for Enhanced Single-Image Super-Resolution with Reduced Computational Requirements"**.

[中文说明 / Chinese README](README.zh-CN.md)

This repository turns EPNet into a complete demo-ready solution asset:

- PyTorch EPNet model and paper-grounded implementation notes
- training, evaluation, and inference CLIs
- FastAPI backend with structured telemetry and artifact serving
- animated Next.js frontend for interactive super-resolution demos
- SQLite usage analytics with replayable history
- Dockerized local stack
- smoke, API, and browser automation coverage

## What This Demo Now Shows

The current demo is designed to feel like a solution engineering artifact rather than a raw research repo.

- bilingual UI toggle for English and Simplified Chinese
- drag-and-drop single-image inference
- batch inference mode with queue and aggregate stats
- A/B compare across `EPNet`, `Bicubic`, and `Baseline`
- x2/x3/x4 scale controls
- checkpoint switcher for any checkpoints found in `checkpoints/*.pt`
- output format switch for `PNG`, `JPEG`, `WEBP`, and `BMP`
- optional tile inference for memory-friendly large-image demos
- session ID support for usage grouping
- downloadable outputs and shareable history links
- replayable inference history from analytics records
- model explanation page with PFEM / ESPM / reconstruction talk track
- deployment/version panel with model version, checkpoint source, build time, git commit, and device target
- one-click reproducibility command cards for train / eval / infer
- analytics cards including request count, session count, average latency, p50 latency, and p95 latency

## Current Status

- Core EPNet model implemented in PyTorch.
- Paper defaults captured in configuration and training CLI.
- Training CLI now supports `cpu`, `cuda`, and `mps`, plus inference-ready EMA checkpoint export.
- Evaluation pipeline reports PSNR and SSIM.
- FastAPI exposes health, model info, single inference, batch inference, usage summary, recent events, and replay history.
- Output artifacts are returned by URL rather than inline base64 payloads.
- Frontend includes animated compare, zoom loupe, pipeline timeline, usage charts, batch queue, and model explainer.
- Playwright smoke coverage exercises upload success, friendly errors, compare slider interaction, refresh stability, and batch mode.
- Docker Compose runs frontend and backend together for local demo use.

## Important Reproduction Note

The repository now prefers the trained deployment artifact at `outputs/run_x4_edge_default/inference.pt` when it exists. The older `checkpoints/demo_x4.pt` checkpoint is kept only as a bootstrap fallback for machines that have not run full training yet.

Every paper ambiguity is documented in [docs/epnet_assumptions.md](/Users/macbook/Desktop/epnet/docs/epnet_assumptions.md).

## Project Structure

- `src/epnet/models`: EPNet model, presets, registry, and block-level modules
- `src/epnet/data`: synthetic, DIV2K, benchmark, transforms, paired datasets, and data builders
- `src/epnet/utils`: device, metrics, manifest, seed, and path helpers
- `src/epnet/train.py`: config-driven training plus legacy-compatible smoke path
- `src/epnet/evaluate.py`: single-folder and multi-dataset evaluation
- `src/epnet/ablate.py`: ablation runner
- `src/epnet/profile.py`: profile reporting
- `src/epnet/export.py`: ONNX export smoke path
- `src/epnet_api`: FastAPI app, schemas, services, and analytics database
- `frontend`: Next.js + TypeScript + Tailwind + Framer Motion + Recharts UI
- `configs`: model, data, and train YAMLs
- `scripts`: dataset setup and local full-training entrypoints
- `tests`: backend, model, config, export, and training smoke coverage
- `docs`: assumptions, audit, dataset setup, edge preset notes, and workflow docs

## Paper-Faithful Scope

- shallow `3x3` convolution for base feature extraction
- PFEM branch with LFEB, modified Swin Transformer, and ESAB
- ESPM branch with DCAB-style channel splitting and pyramid fusion
- reconstruction by `3x3` convolution plus `PixelShuffle`
- default `n = 4` PFEM blocks
- appendix-aligned training defaults:
  - patch size `48`
  - batch size `32`
  - Adam `lr=5e-4`, betas `(0.9, 0.99)`
  - L1 loss
  - EMA decay `0.999`
  - `1e6` iterations
  - no warm-up

## Setup

### Backend

```bash
python3 -m venv .venv
.venv/bin/pip install '.[dev]'
```

### Frontend

```bash
cd frontend
npm install
cd ..
```

## Synthetic Smoke Path

```bash
PYTHONPATH=src .venv/bin/python -m epnet.train \
  --model-config configs/model/edge_default_x2.yaml \
  --data-config configs/data/synthetic_x2.yaml \
  --train-config configs/train/smoke.yaml

PYTHONPATH=src .venv/bin/python -m epnet.experiments \
  --output-dir outputs/model_ablation_mps \
  --report-path docs/model_ablation_results.md \
  --json-path docs/model_ablation_results.json \
  --device mps \
  --scale 2 \
  --steps 8 \
  --synthetic-count 128
```

Synthetic training and ablation are now treated as smoke/regression paths. They remain useful for CI-friendly checks and local sanity verification, but they are no longer the mainline training story for this repository.

## Real-Data Full Training Path

### 1. Download datasets

```bash
PYTHONPATH=src .venv/bin/python scripts/download_real_data.py --data-root data
```

### 2. Prepare bicubic LR caches

```bash
PYTHONPATH=src .venv/bin/python scripts/prepare_real_data.py --data-root data --scales 2 3 4
```

### 3. Run full local x4 training

```bash
PYTHONPATH=src .venv/bin/python scripts/run_local_full_x4.py
```

### 4. Resume full x4 training

`scripts/run_local_full_x4.py` automatically resumes from `outputs/run_x4_edge_default/latest.pt` when that file already exists.

### 5. Evaluate a checkpoint

```bash
PYTHONPATH=src .venv/bin/python -m epnet.evaluate \
  --checkpoint outputs/run_x4_edge_default/inference.pt \
  --data-config configs/data/div2k_x4.yaml \
  --output-json outputs/run_x4_edge_default/eval.json \
  --output-markdown outputs/run_x4_edge_default/eval.md
```

## Config-Driven Training System

The repository now supports config-driven training:

```bash
PYTHONPATH=src .venv/bin/python -m epnet.train \
  --model-config configs/model/edge_default.yaml \
  --data-config configs/data/div2k_x4.yaml \
  --train-config configs/train/local_full_x4.yaml
```

- automatic device selection across `cuda`, `mps`, and `cpu`
- optional CUDA AMP with safe fallback on MPS/CPU
- config snapshot saved into each run directory
- reproducible `manifest.json` with git commit, command, seed, device, dataset paths, and config payload
- EMA tracking during training
- resume support with optimizer and scaler restoration
- eval during training through the primary validation directory
- best checkpoint + latest checkpoint
- rolling snapshots
- dedicated inference checkpoint export using EMA weights
- evaluation and profile report generation after config-driven runs

Generated run artifacts:

- `outputs/<run_name>/manifest.json`
- `outputs/<run_name>/train_log.jsonl`
- `outputs/<run_name>/latest.pt`
- `outputs/<run_name>/best.pt`
- `outputs/<run_name>/inference.pt`
- `outputs/<run_name>/eval.json`
- `outputs/<run_name>/eval.md`
- `outputs/<run_name>/profile.json`
- `outputs/<run_name>/profile.md`
- `outputs/<run_name>/model.onnx`

### Model Variants

- `paper_like`: baseline comparison preset closest to the paper-grounded configuration
- `edge_tiny`: smallest edge-oriented preset
- `edge_default`: recommended mainline preset, using PFEM depth `2` and ESPM levels `2`
- `balanced_quality`: wider preset for local quality-oriented comparisons

Legacy aliases `tiny`, `paper`, and `balanced` are still accepted for compatibility, but the repo now uses the new preset names in configs and docs.

### Variant Comparison And Ablation

The repo now includes a reproducible ablation runner for model-only comparisons.

- command entry: `python -m epnet.experiments`
- default suite: `tiny`, `paper`, `balanced`, plus `paper` ablations for PFEM depth, shared PFEM weights, and ESPM depth
- outputs:
  - raw checkpoints in `outputs/model_ablation_mps/`
  - markdown summary in `docs/model_ablation_results.md`
  - raw metrics in `docs/model_ablation_results.json`

The checked-in report documents one real local MPS run on deterministic synthetic data. Treat it as an engineering comparison, not a paper benchmark.

## Dataset Download And Preparation

- dataset setup guide: [docs/dataset_setup.md](/Users/macbook/Desktop/epnet/docs/dataset_setup.md)
- real training workflow: [docs/real_training_workflow.md](/Users/macbook/Desktop/epnet/docs/real_training_workflow.md)
- edge preset notes: [docs/edge_preset_notes.md](/Users/macbook/Desktop/epnet/docs/edge_preset_notes.md)

The download script supports:

- DIV2K train and validation HR
- Set5 / Set14 / BSD100 / Urban100 via the VDSR benchmark test pack

Manual fallback:

- Manga109 is not auto-downloaded by default; place it manually under `data/benchmarks/Manga109/HR`

## Output Directory Structure

```text
outputs/
  run_x4_edge_default/
    config_snapshot/
      model.json
      data.json
      train.json
    manifest.json
    train_log.jsonl
    latest.pt
    best.pt
    inference.pt
    eval.json
    eval.md
    profile.json
    profile.md
    model.onnx
```

## Known Limitations

- The demo checkpoint in `checkpoints/demo_x4.pt` is still a bootstrap artifact, not a full benchmark-trained weight.
- Synthetic ablation results are engineering comparisons, not paper claims.
- Manga109 setup is manual unless you already have approved access.
- ONNX export is exercised as a smoke path and may report unsupported ops depending on the active PyTorch exporter/runtime.

## One-Command Local Demo

```bash
./scripts/run_local.sh
```

This script will:

- create `.venv` if missing
- install backend dependencies
- install frontend dependencies
- prefer `outputs/run_x4_edge_default/inference.pt` for deployment when it exists
- fall back to `checkpoints/demo_x4.pt` only when no trained deployment checkpoint is available
- start FastAPI on `http://localhost:8000`
- start Next.js on `http://localhost:3000`

If your local ports are already occupied, override them without changing code:

```bash
EPNET_API_PORT=8011 EPNET_FRONTEND_PORT=3011 ./scripts/run_local.sh
```

## API Summary

Primary endpoints:

- `GET /api/v1/health`
- `GET /api/v1/model/info`
- `POST /api/v1/infer`
- `POST /api/v1/infer/batch`
- `GET /api/v1/usage/summary`
- `GET /api/v1/usage/recent`
- `GET /api/v1/history/{request_id}`

Backward-compatible aliases:

- `POST /api/v1/super-resolve`
- `GET /api/v1/analytics/summary`

### Single Inference Options

`POST /api/v1/infer` accepts multipart form data:

- `file`
- `session_id`
- `method`: `epnet`, `bicubic`, `baseline`
- `scale`: `2`, `3`, `4`
- `output_format`: `PNG`, `JPEG`, `WEBP`, `BMP`
- `tile_size`: `0`, `256`, `384`, `512`
- `checkpoint_name`: optional, only for EPNet mode

### Batch Inference

`POST /api/v1/infer/batch` accepts repeated `files` fields plus the same inference options as single-image mode.

### Analytics and Replay

Each inference record stores:

- request ID and UTC timestamp
- session ID
- input/output resolution
- input/output bytes
- selected method and scale
- checkpoint name
- output format
- tile size
- latency
- parameter count
- estimated MACs/FLOPs
- artifact URLs for replay

## Checkpoints and Deployment Artifact Selection

By default, the backend now serves a single explicit deployment artifact:

- primary default: `outputs/run_x4_edge_default/inference.pt`
- fallback default: `checkpoints/demo_x4.pt`

The deployment backend is explicit:

- default runtime backend: `pytorch`
- optional backend selector: `EPNET_INFERENCE_BACKEND=pytorch|onnx`

To swap the deployed checkpoint:

```bash
EPNET_CHECKPOINT_PATH=/absolute/path/to/inference.pt ./scripts/run_local.sh
```

If you want the frontend checkpoint switcher to expose a directory of multiple `.pt` artifacts, point the backend at that directory explicitly:

```bash
EPNET_CHECKPOINT_DIR=/absolute/path/to/checkpoint_dir ./scripts/run_local.sh
```

If `EPNET_INFERENCE_BACKEND=onnx` is requested, the system expects a valid ONNX export and a working `onnxruntime` installation. The current deployment default remains PyTorch because it is the most robust path for the trained EPNet artifact.

## Environment Variables

- `EPNET_CHECKPOINT_PATH`
- `EPNET_CHECKPOINT_DIR`
- `EPNET_INFERENCE_BACKEND`
- `EPNET_ONNX_MODEL_PATH`
- `EPNET_ANALYTICS_DB_PATH`
- `EPNET_ARTIFACTS_DIR`
- `EPNET_MAX_UPLOAD_BYTES`
- `EPNET_MAX_IMAGE_PIXELS`
- `EPNET_CORS_ORIGINS`
- `EPNET_BUILD_TIME`
- `EPNET_GIT_COMMIT`
- `EPNET_PROFILE_INPUT_SIZE`

Useful script-level overrides:

- `EPNET_API_HOST`
- `EPNET_API_PORT`
- `EPNET_FRONTEND_HOST`
- `EPNET_FRONTEND_PORT`
- `EPNET_API_RELOAD`

## Docker

```bash
docker compose up --build
```

- frontend: [http://localhost:3000](http://localhost:3000)
- backend: [http://localhost:8000](http://localhost:8000)

Notes:

- backend Docker uses CPU-only PyTorch wheels for lighter local demo builds
- frontend build script clears `.next` before production build to avoid stale app-router build artifacts
- compose includes health checks for both services

## Verification

Validated recently with:

- `PYTHONPATH=src .venv/bin/ruff check src tests`
- `PYTHONPATH=src .venv/bin/mypy src`
- `PYTHONPATH=src .venv/bin/pytest`
- `cd frontend && npm run lint`
- `cd frontend && npm run typecheck`
- `cd frontend && npm run build`
- `cd frontend && npm run test:e2e`
- `docker compose config`
- `docker compose up -d --build`

## Demo Talking Points

For a presentation or interview, the strongest story is usually:

1. EPNet as an efficient SR architecture, not just a quality-max model
2. system flow from upload to analytics logging
3. deployment metadata and checkpoint provenance
4. reproducibility from UI back to CLI
5. edge/deployment readiness via batching, tile mode, replay, and Docker

## Related Docs

- [docs/epnet_assumptions.md](/Users/macbook/Desktop/epnet/docs/epnet_assumptions.md)
- [docs/final_audit_report.md](/Users/macbook/Desktop/epnet/docs/final_audit_report.md)
- [docs/optimization_report.md](/Users/macbook/Desktop/epnet/docs/optimization_report.md)
- [docs/demo_checklist.md](/Users/macbook/Desktop/epnet/docs/demo_checklist.md)
