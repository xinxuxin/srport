# System Deployment Report

## Deployment Decision

The system now uses the trained deployment checkpoint as the primary runtime artifact:

- default deployment artifact: `outputs/run_x4_edge_default/inference.pt`
- default runtime backend: `pytorch`

This was chosen over ONNX as the default path because:

- the PyTorch inference path is already integrated into the FastAPI service and CLI
- the trained `inference.pt` artifact is explicitly exported for runtime use
- ONNX export succeeds, but the current model still emits trace warnings from the windowed global-context block
- `onnxruntime` is not installed in the default local environment, so ONNX is better treated as an optional backend rather than the default deployment path

## Deployment Configuration

Supported runtime configuration:

- `EPNET_CHECKPOINT_PATH`: explicit `.pt` deployment artifact
- `EPNET_CHECKPOINT_DIR`: optional directory of multiple `.pt` artifacts for the frontend checkpoint selector
- `EPNET_INFERENCE_BACKEND`: `pytorch` or `onnx`
- `EPNET_ONNX_MODEL_PATH`: explicit ONNX artifact path when experimenting with ONNX runtime
- `EPNET_API_PORT`, `EPNET_FRONTEND_PORT`: one-command demo port overrides

## What Was Integrated

The following system-level integration work was completed:

- the API default checkpoint path now prefers `outputs/run_x4_edge_default/inference.pt`
- the deployment backend is now explicit in settings and model metadata
- the API now exposes deployment metadata for:
  - runtime backend
  - deployed artifact path
- the backend no longer silently falls back to a random untrained runtime when a configured artifact is missing
- `scripts/run_local.sh` now prefers the trained deployment artifact automatically
- `scripts/run_local.sh` now supports port overrides for easier local demos
- `scripts/run_local.sh` now disables backend reload by default for more stable demo behavior

## End-to-End Validation Performed

### System startup

The full system was started successfully with:

```bash
EPNET_API_PORT=8011 EPNET_FRONTEND_PORT=3011 ./scripts/run_local.sh
```

Validated startup results:

- FastAPI reachable at `http://localhost:8011`
- Next.js reachable at `http://localhost:3011`
- backend reported the deployment checkpoint:
  - `/Users/macbook/Desktop/epnet/outputs/run_x4_edge_default/inference.pt`

### Model metadata

Validated:

```bash
curl http://localhost:8011/api/v1/model/info
```

Confirmed fields:

- `checkpoint_path` points to `outputs/run_x4_edge_default/inference.pt`
- `deployment.runtime_backend` is `pytorch`
- `deployment.artifact_path` points to the trained deployment checkpoint

### Real inference

Validated:

```bash
curl -X POST http://localhost:8011/api/v1/infer \
  -F 'file=@data/samples/demo_input.png;type=image/png' \
  -F 'session_id=deployment-e2e' \
  -F 'method=epnet' \
  -F 'scale=4' \
  -F 'output_format=PNG' \
  -F 'tile_size=0'
```

Observed result:

- input resolution: `64x64`
- output resolution: `256x256`
- runtime latency returned by the API
- output artifact URL returned successfully
- output artifact retrieved successfully from `/artifacts/...`

### Failure cases

Validated:

- invalid upload file returns `400` with a friendly message
- missing checkpoint path raises a clear startup error
- `EPNET_INFERENCE_BACKEND=onnx` currently raises a clear startup error when `onnxruntime` is not installed

## One-Command Demo Path

Start the system:

```bash
./scripts/run_local.sh
```

If ports are already occupied:

```bash
EPNET_API_PORT=8011 EPNET_FRONTEND_PORT=3011 ./scripts/run_local.sh
```

Send one image through the API:

```bash
curl -X POST http://localhost:8011/api/v1/infer \
  -F 'file=@data/samples/demo_input.png;type=image/png' \
  -F 'session_id=deployment-e2e' \
  -F 'method=epnet' \
  -F 'scale=4' \
  -F 'output_format=PNG' \
  -F 'tile_size=0'
```

Inspect the result:

- frontend UI: `http://localhost:3011`
- backend model metadata: `http://localhost:8011/api/v1/model/info`
- output artifact URL: returned in the inference response payload

## Known Deployment Limitations

- ONNX export exists and passes smoke export, but ONNX runtime inference is not the default deployment path in this environment
- ONNX export still emits trace warnings from the Swin-style windowed global-context block, so it should be treated as shape-sensitive until a full ONNX runtime validation pass is completed
- the default deployment backend remains PyTorch for robustness

## Actual Outcome

- deployment artifact selection: `PASS`
- model loading in system: `PASS`
- end-to-end inference path: `PASS`
- user-facing system integration: `PASS`
- one-command run path: `PASS`
- docs updated: `PASS`
