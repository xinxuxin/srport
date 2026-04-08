# Demo Checklist

## Pre-demo

- Confirm `checkpoints/demo_x4.pt` exists or run `./scripts/run_local.sh` once to bootstrap it.
- Start the stack with `./scripts/run_local.sh`.
- Verify backend health at `http://localhost:8000/api/v1/health`.
- Verify frontend loads at `http://localhost:3000`.

## Happy-path demo

- Upload a PNG or JPEG image.
- Confirm the result panel updates with a super-resolved image.
- Confirm the compare slider moves without layout breakage.
- Confirm the stats cards show:
  - model name
  - parameter count
  - latency
  - input resolution
  - output resolution
  - FLOPs / MACs
- Confirm the usage dashboard increments request count.

## Error-handling demo

- Upload a non-image file and confirm a friendly error appears.
- Upload an oversized file and confirm the frontend or backend rejects it cleanly.
- Temporarily point `EPNET_CHECKPOINT_PATH` at a missing file and confirm the backend reports random-initialization fallback instead of crashing.

## Verification commands

- Backend/tests:
  - `PYTHONPATH=src .venv/bin/pytest`
  - `PYTHONPATH=src .venv/bin/mypy src`
  - `.venv/bin/ruff check src tests`
- Frontend:
  - `cd frontend && npm run lint`
  - `cd frontend && npm run typecheck`
  - `cd frontend && npm run build`
- Docker config:
  - `docker compose config`
  - `docker compose up -d --build`
  - `docker compose ps`

## Known constraints

- The included demo checkpoint is a lightweight synthetic bootstrap checkpoint for local demos, not a full benchmark-trained paper reproduction artifact.
- Browser interaction is still verified manually; there is no Playwright-style automated UI suite yet.
