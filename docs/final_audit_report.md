# Final Audit Report

## Scope

This audit covered the EPNet single-image super-resolution demo across:

- PyTorch model correctness and shape behavior
- training, evaluation, inference, and checkpoint flows
- FastAPI backend routes and upload handling
- Next.js frontend build and runtime behavior
- usage logging and dashboard data flow
- local packaging and Docker configuration

## What Was Tested

### Static checks

- `ruff` on `src` and `tests`
- `mypy` on `src`
- frontend lint
- frontend typecheck
- frontend production build

### Model and pipeline checks

- forward-pass smoke tests for x2, x3, and x4
- NaN check on outputs
- parameter-count and MAC/FLOP profiling smoke test
- checkpoint load path
- CPU inference path
- tiny synthetic training run
- checkpoint save and resume
- evaluation pass with PSNR and SSIM output

### Backend checks

- app import and startup smoke test
- live `GET /api/v1/health`
- live `GET /api/v1/model/info`
- live `GET /api/v1/usage/summary`
- live `GET /api/v1/usage/recent`
- live `POST /api/v1/infer` with a valid image
- live invalid text upload
- live corrupt `image/png` upload
- repeated inference calls and analytics growth
- missing-checkpoint fallback behavior

### Frontend checks

- homepage load from a live Next.js server
- presence of main EPNet demo UI content in the served page
- integration against the live backend
- drag-and-drop/upload code path review
- production build and route generation

### Docker checks

- `docker compose config`
- full `docker compose up -d --build`
- `docker compose ps`
- backend and frontend container health status
- live backend endpoint checks through the containerized stack
- live frontend homepage load through the containerized stack

## What Failed Initially

- Corrupt uploads labeled as `image/png` raised a server-side `500` instead of returning a friendly client error.
- The backend route surface did not match the expected demo contract. `/infer`, `/usage/summary`, and `/usage/recent` were missing.
- Request latency was measured by profiling the model on every request and then running inference again, which both inflated latency and doubled the compute work.
- The frontend upload flow leaked object URLs across repeated uploads.
- The frontend typecheck path was brittle because it depended on Next-generated route types already existing.
- The synthetic training fallback had a shape mismatch path, and training config scale could silently diverge from model upscale.
- Test coverage did not cover alias routes, corrupt-image uploads, x2/x3 shapes, oversized uploads, or resume behavior.
- Docker builds pulled large CUDA-oriented PyTorch dependencies in the backend image, which made local demo builds unnecessarily heavy.
- The frontend Docker image used a single-stage build and lacked reliable runtime health checks.

## What Was Fixed

### Backend

- Added `/api/v1/infer`, `/api/v1/usage/summary`, and `/api/v1/usage/recent` while keeping existing aliases.
- Added upload guardrails:
  - explicit allowed MIME types
  - upload byte limit
  - decoded pixel limit
- Added graceful invalid-image handling so corrupt image bytes return `400`.
- Switched request-time inference execution to `torch.inference_mode()`.
- Replaced per-request model profiling with real wall-clock latency measurement for the actual inference pass.
- Added analytics `recent()` support and separated usage-summary and recent-usage retrieval.

### Model and training

- Added x2/x3/x4 shape coverage.
- Added resume support to training checkpoints, including optimizer-state persistence.
- Added a fast-fail consistency check so `TrainConfig.scale` must match `ModelConfig.upscale`.
- Fixed the synthetic training fallback to generate HR/LR pairs with the correct patch geometry.

### Frontend

- Updated the frontend API client to use the new primary endpoints.
- Improved API error parsing for friendlier user-facing errors.
- Added client-side upload validation for type and size.
- Added object-URL cleanup to avoid preview-memory leaks.
- Added a lightweight comparison slider to make before/after inspection clearer.
- Made `npm run typecheck` stable by generating Next route types first.

### Tests

- Added tests for:
  - x2/x3/x4 model output shapes
  - `/infer` alias
  - `/usage/summary`
  - `/usage/recent`
  - corrupt-image upload handling
  - oversized upload rejection
  - checkpoint resume behavior

### Docker and packaging

- Switched the backend Docker image to an explicit CPU-only PyTorch install path for local demo builds.
- Added backend and frontend container health checks and verified both services reach `healthy` in `docker compose`.
- Fixed the frontend runtime container to bind to `0.0.0.0`, which made the internal health probe reliable.
- Converted the frontend Docker image to a multi-stage Next.js standalone build, reducing the runtime image size substantially.
- Completed live Docker-backed verification of:
  - `GET /api/v1/health`
  - `GET /api/v1/model/info`
  - `GET /api/v1/usage/summary`
  - `GET /api/v1/usage/recent`
  - `POST /api/v1/infer` with valid, invalid, corrupt, and oversized uploads
  - repeated inference calls with analytics growth

## Remaining Gaps / Future Improvements

- The frontend lint command still uses legacy ESLint config compatibility mode and emits a deprecation warning.
- The demo still returns SR images inline as base64 JSON, which is simple but inefficient for larger production-style payloads.
- The project still relies on a small synthetic demo checkpoint for easy local bring-up; a true paper-style trained checkpoint remains a separate longer-running step.
- There is no browser-automation E2E suite yet; live HTTP verification was completed, but UI interaction was not automated in-browser in this environment.
