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
- Evaluation pipeline reports PSNR and SSIM.
- FastAPI exposes health, model info, single inference, batch inference, usage summary, recent events, and replay history.
- Output artifacts are returned by URL rather than inline base64 payloads.
- Frontend includes animated compare, zoom loupe, pipeline timeline, usage charts, batch queue, and model explainer.
- Playwright smoke coverage exercises upload success, friendly errors, compare slider interaction, refresh stability, and batch mode.
- Docker Compose runs frontend and backend together for local demo use.

## Important Reproduction Note

The default `demo_x4.pt` checkpoint is a demo bootstrap artifact, not a fully paper-trained benchmark checkpoint. The repository is runnable end to end, but if you want stronger visual quality for a formal benchmark or interview demo, train a real checkpoint and place it in `checkpoints/`.

Every paper ambiguity is documented in [docs/epnet_assumptions.md](/Users/macbook/Desktop/epnet/docs/epnet_assumptions.md).

## Project Structure

- `src/epnet`: model, modules, datasets, metrics, profiling, and train/eval/infer CLIs
- `src/epnet_api`: FastAPI app, schemas, services, and analytics database
- `frontend`: Next.js + TypeScript + Tailwind + Framer Motion + Recharts UI
- `tests`: backend, model, and training smoke coverage
- `docs`: assumptions, audit, optimization, and demo documentation
- `scripts`: local developer helpers
- `outputs`: generated analytics DB and output artifacts at runtime

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

## Core Commands

```bash
PYTHONPATH=src epnet-train --output checkpoints/epnet_x4.pt --scale 4 --train-dir path/to/div2k_train_hr
PYTHONPATH=src epnet-train --output checkpoints/epnet_x4.pt --scale 4 --train-dir path/to/div2k_train_hr --resume checkpoints/epnet_x4.pt
PYTHONPATH=src epnet-eval --checkpoint checkpoints/epnet_x4.pt --hr-dir path/to/benchmark_hr
PYTHONPATH=src epnet-infer --checkpoint checkpoints/epnet_x4.pt --input data/samples/demo_input.png --output outputs/demo_output.png
PYTHONPATH=src .venv/bin/uvicorn epnet_api.app.main:app --reload
cd frontend && npm run dev
./scripts/run_local.sh
```

## One-Command Local Demo

```bash
./scripts/run_local.sh
```

This script will:

- create `.venv` if missing
- install backend dependencies
- install frontend dependencies
- bootstrap `checkpoints/demo_x4.pt` if missing
- start FastAPI on `http://localhost:8000`
- start Next.js on `http://localhost:3000`

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

## Checkpoints and Scale Switching

The backend automatically scans `checkpoints/*.pt`.

- Any discovered checkpoint becomes selectable in the frontend.
- EPNet scale options depend on the loaded checkpoints.
- Baseline methods remain available for x2/x3/x4 even if EPNet checkpoints do not exist for all scales.

If you add multiple checkpoints such as:

- `checkpoints/demo_x2.pt`
- `checkpoints/demo_x3.pt`
- `checkpoints/demo_x4.pt`
- `checkpoints/epnet_stage200k_x4.pt`

the frontend checkpoint switcher will expose them automatically.

## Environment Variables

- `EPNET_CHECKPOINT_PATH`
- `EPNET_ANALYTICS_DB_PATH`
- `EPNET_ARTIFACTS_DIR`
- `EPNET_MAX_UPLOAD_BYTES`
- `EPNET_MAX_IMAGE_PIXELS`
- `EPNET_CORS_ORIGINS`
- `EPNET_BUILD_TIME`
- `EPNET_GIT_COMMIT`
- `EPNET_PROFILE_INPUT_SIZE`

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
