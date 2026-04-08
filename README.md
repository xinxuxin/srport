# EPNet Demo

Production-oriented single-image super-resolution demo based on the paper **"EPNet: An Efficient Pyramid Network for Enhanced Single-Image Super-Resolution with Reduced Computational Requirements"**.

This repository reproduces the paper's EPNet architecture as faithfully as possible from the published text and figures, documents every underspecified detail, provides PyTorch training and evaluation pipelines, and exposes inference through a FastAPI backend plus a polished Next.js frontend.

## Current Status

- Core EPNet model implemented in PyTorch.
- Paper defaults captured in configuration and training CLI.
- Evaluation pipeline with PSNR and SSIM.
- Inference CLI with parameter count, estimated MACs/FLOPs, and latency profiling.
- FastAPI backend with `/health`, `/model/info`, `/super-resolve`, and `/analytics/summary`.
- SQLite usage analytics for inference requests.
- Animated Next.js frontend with drag-and-drop upload, result previews, telemetry cards, and charts.
- Dockerfiles, `docker-compose.yml`, and a one-command local launcher.
- Explicit reproduction notes in [docs/epnet_assumptions.md](/Users/macbook/Desktop/epnet/docs/epnet_assumptions.md).

## Project Structure

- `src/epnet`: EPNet model, modules, datasets, metrics, profiling, training, evaluation, and inference CLIs.
- `src/epnet_api`: FastAPI application package.
- `tests`: Core smoke and metric tests.
- `docs`: Reproduction assumptions and implementation notes.
- `frontend`: Next.js web application.
- `scripts`: Local developer workflow helpers.

## Paper-Faithful Scope

- Shallow `3x3` convolution for base feature extraction.
- PFEM branch with LFEB, modified Swin Transformer, and ESAB.
- ESPM branch with DCAB-style channel splitting and pyramid fusion.
- Reconstruction by `3x3` convolution plus `PixelShuffle`.
- Default `n = 4` PFEM submodules.
- Training defaults aligned with the appendix:
  - patch size `48`
  - batch size `32`
  - Adam `lr=5e-4`, betas `(0.9, 0.99)`
  - L1 loss
  - EMA decay `0.999`
  - `1e6` iterations
  - no warm-up

## Core Commands

These work once the Python dependencies are installed:

```bash
python3 -m venv .venv
.venv/bin/pip install '.[dev]'
cd frontend && npm install && cd ..
PYTHONPATH=src epnet-train --output checkpoints/epnet_x4.pt --train-dir path/to/div2k_train_hr
PYTHONPATH=src epnet-eval --checkpoint checkpoints/epnet_x4.pt --hr-dir path/to/benchmark_hr
PYTHONPATH=src epnet-infer --checkpoint checkpoints/epnet_x4.pt --input input.png --output output.png
PYTHONPATH=src .venv/bin/uvicorn epnet_api.app.main:app --reload
./scripts/run_local.sh
```

## Local Demo

Run the whole stack locally with:

```bash
./scripts/run_local.sh
```

This script will:

- create a local virtual environment if needed
- install backend dependencies
- install frontend dependencies
- bootstrap a small synthetic demo checkpoint if `checkpoints/demo_x4.pt` is missing
- start FastAPI on `http://localhost:8000`
- start Next.js on `http://localhost:3000`

## Docker

Build and run the demo stack with:

```bash
docker compose up --build
```

The frontend will be available on `http://localhost:3000` and the backend on `http://localhost:8000`.

## Verification So Far

- `PYTHONPATH=src PYTHONPYCACHEPREFIX=.pycache python3 -m compileall src`
- `PYTHONPATH=src python3` smoke tests for model construction and forward pass
- `.venv/bin/ruff check src tests`
- `PYTHONPATH=src .venv/bin/pytest`
- `PYTHONPATH=src .venv/bin/mypy src`
- `cd frontend && npm run lint`
- `cd frontend && npm run typecheck`
- `cd frontend && npm run build`

## Reproduction Notes

The paper does not publish enough tensor-level detail to uniquely recover every block. Every non-explicit design choice is tracked in [docs/epnet_assumptions.md](/Users/macbook/Desktop/epnet/docs/epnet_assumptions.md), separated into:

- paper-derived implementation details
- engineering assumptions
- optional demo improvements
