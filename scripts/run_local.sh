#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi

.venv/bin/pip install '.[dev]'

if [[ ! -d "frontend/node_modules" ]]; then
  (cd frontend && npm install)
fi

if [[ ! -f "checkpoints/demo_x4.pt" ]]; then
  echo "Bootstrapping demo checkpoint..."
  PYTHONPATH=src .venv/bin/epnet-train \
    --output checkpoints/demo_x4.pt \
    --steps 60 \
    --batch-size 8 \
    --patch-size 48 \
    --synthetic-count 256
fi

cleanup() {
  jobs -p | xargs -r kill
}

trap cleanup EXIT INT TERM

PYTHONPATH=src .venv/bin/uvicorn epnet_api.app.main:app \
  --reload \
  --host 0.0.0.0 \
  --port 8000 &

(cd frontend && NEXT_PUBLIC_API_BASE_URL=http://localhost:8000/api/v1 npm run dev -- --hostname 0.0.0.0 --port 3000) &

wait -n
