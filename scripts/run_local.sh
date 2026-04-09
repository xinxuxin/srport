#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TRAINED_CHECKPOINT="outputs/run_x4_edge_default/inference.pt"
TRAINED_ONNX="outputs/run_x4_edge_default/model.onnx"
DEMO_CHECKPOINT="checkpoints/demo_x4.pt"
API_HOST="${EPNET_API_HOST:-0.0.0.0}"
API_PORT="${EPNET_API_PORT:-8000}"
FRONTEND_HOST="${EPNET_FRONTEND_HOST:-0.0.0.0}"
FRONTEND_PORT="${EPNET_FRONTEND_PORT:-3000}"

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

if [[ -z "${EPNET_CHECKPOINT_PATH:-}" ]]; then
  if [[ -f "$TRAINED_CHECKPOINT" ]]; then
    export EPNET_CHECKPOINT_PATH="$ROOT_DIR/$TRAINED_CHECKPOINT"
  else
    export EPNET_CHECKPOINT_PATH="$ROOT_DIR/$DEMO_CHECKPOINT"
  fi
fi

if [[ -z "${EPNET_INFERENCE_BACKEND:-}" ]]; then
  export EPNET_INFERENCE_BACKEND="pytorch"
fi

if [[ -z "${EPNET_ONNX_MODEL_PATH:-}" && -f "$TRAINED_ONNX" ]]; then
  export EPNET_ONNX_MODEL_PATH="$ROOT_DIR/$TRAINED_ONNX"
fi

if [[ -z "${EPNET_CORS_ORIGINS:-}" ]]; then
  export EPNET_CORS_ORIGINS="http://localhost:${FRONTEND_PORT},http://127.0.0.1:${FRONTEND_PORT}"
fi

echo "Using deployment checkpoint: $EPNET_CHECKPOINT_PATH"
echo "Using inference backend: $EPNET_INFERENCE_BACKEND"
echo "Backend URL: http://localhost:${API_PORT}"
echo "Frontend URL: http://localhost:${FRONTEND_PORT}"
if [[ "${EPNET_API_RELOAD:-0}" == "1" ]]; then
  echo "Backend reload mode: enabled"
else
  echo "Backend reload mode: disabled"
fi

cleanup() {
  jobs -p | xargs -r kill
}

trap cleanup EXIT INT TERM

uvicorn_args=(
  --host "$API_HOST"
  --port "$API_PORT"
)
if [[ "${EPNET_API_RELOAD:-0}" == "1" ]]; then
  uvicorn_args=(--reload "${uvicorn_args[@]}")
fi

PYTHONPATH=src .venv/bin/uvicorn epnet_api.app.main:app "${uvicorn_args[@]}" &
backend_pid=$!

(cd frontend && NEXT_PUBLIC_API_BASE_URL="http://localhost:${API_PORT}/api/v1" npm run dev -- --hostname "$FRONTEND_HOST" --port "$FRONTEND_PORT") &
frontend_pid=$!

while kill -0 "$backend_pid" 2>/dev/null && kill -0 "$frontend_pid" 2>/dev/null; do
  sleep 1
done

wait "$backend_pid" || true
wait "$frontend_pid" || true
