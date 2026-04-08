# Optimization Report

## Implemented Optimizations

### Inference path

- Removed per-request `profile_model(...)` calls from live inference requests.
  - Before: every request profiled the model and then ran inference again.
  - After: live latency is measured around the actual inference pass only.
- Switched request inference to `torch.inference_mode()` to reduce autograd overhead.
- Switched CLI evaluation and inference to `torch.inference_mode()` as well.
- Kept model-info profiling cached at startup, so parameter count and reference MAC/FLOP metadata remain available without repeated recomputation.
- Added a shared checkpoint-loading helper that prefers safer `weights_only=True` loading where supported.

### Upload and decode path

- Added early upload-size rejection before image decode.
- Added decoded pixel-count guard to avoid oversized image processing.
- Added format normalization and invalid-image short-circuiting to avoid wasted downstream work.

### Frontend

- Added object-URL cleanup so repeated uploads do not accumulate preview-memory leaks.
- Added client-side file validation so obviously invalid or oversized files are rejected before an API round trip.
- Stabilized type generation so frontend typecheck no longer depends on a prior build.
- Converted the frontend Docker image to a multi-stage standalone Next.js runtime.
  - Result: `epnet-frontend` dropped from about `1.45 GB` to about `302 MB` in local verification.

### Docker / packaging

- Switched backend Docker installs to CPU-only PyTorch wheels for local demo builds.
  - Result: Docker no longer pulls the large CUDA dependency chain for the backend.
- Added backend and frontend container health checks for better demo observability.
- Fixed the frontend runtime container to bind on `0.0.0.0`, making container health probes reliable.

### Training

- Added explicit scale/upscale consistency validation to fail fast rather than reaching an expensive shape mismatch later in the training loop.
- Added resume support so interrupted training can continue from saved optimizer/model/EMA state.

## Recommended Future Optimizations

### Model / inference

- Add device selection and optional GPU inference path in the CLI and API.
- Add tiled inference for large inputs to reduce memory spikes.
- Consider optional mixed-precision inference on CUDA-capable systems.
- Consider separating reference MAC/FLOP profiling from startup for faster cold starts.

### Backend

- Replace inline base64 image responses with streamed image files or signed asset URLs for larger outputs.
- Add structured error schema models for frontend-friendly handling.
- Add a lightweight process-wide cache for analytics summary if request volume grows.
- Add SQLite indexes if analytics tables are expected to grow substantially.
- Consider splitting model profile metadata generation from API startup if cold-start time becomes important.

### Frontend

- Move from legacy ESLint config compatibility mode to a flat ESLint config.
- Add browser-automated smoke tests for upload, error states, and analytics rendering.
- Consider using `next/image` or a dedicated canvas-based compare component if image sizes increase.
- Consider making the dashboard cards and analytics panel stream or skeleton-render independently to reduce perceived loading delay on first load.

### DevEx

- Add a clean-room verification script that wipes transient artifacts and reruns the full local check set.
- Add a dedicated backend smoke-test script for CI.
- Add environment-variable documentation for all runtime knobs, especially upload and checkpoint settings.
- Consider a slimmer backend base image or wheelhouse caching strategy if Docker rebuild frequency becomes high.
