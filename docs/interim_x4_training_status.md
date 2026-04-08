# Interim x4 Training Status

## Status

- Full completion this session: `NO`
- Current run directory: `outputs/run_x4_edge_default`
- Current latest checkpoint: `outputs/run_x4_edge_default/latest.pt`
- Current best checkpoint: `outputs/run_x4_edge_default/best.pt`

## Current Run State

- Model preset: `edge_default`
- Scale: `x4`
- Train dataset: `DIV2K`
- Device used this session: `mps`
- Configured total steps: `100000`
- Current saved latest step: `1200`
- Current saved best step: `1200`

The run was resumed from the existing mainline checkpoint and continued successfully. Resume was confirmed by log output continuing from step `425` rather than restarting from step `0`.

## Current Best Metrics

### Validation During Training

Current best validation metrics stored in `best.pt` and `latest.pt`:

- PSNR: `21.986848703617575`
- SSIM: `0.4050072133541107`
- Samples: `8`

### Real Benchmark Evaluation

The benchmark evaluation was refreshed for the current `best.pt` and written to:

- `outputs/run_x4_edge_default/eval.json`
- `outputs/run_x4_edge_default/eval.md`

Current benchmark summary:

- Overall mean PSNR: `21.219659680916344`
- Overall mean SSIM: `0.43059884795120784`

Per-dataset means:

- `Set5`: PSNR `22.0710`, SSIM `0.50687`
- `Set14`: PSNR `21.1806`, SSIM `0.41060`
- `BSD100`: PSNR `21.7920`, SSIM `0.42036`
- `Urban100`: PSNR `19.8351`, SSIM `0.38457`

`Manga109` is still not included because it has not been placed locally.

## Current Artifacts

The current run directory contains:

- `manifest.json`
- `train_log.jsonl`
- `latest.pt`
- `best.pt`
- `step1100.pt`
- `step1200.pt`
- `eval.json`
- `eval.md`
- `profile.json`
- `profile.md`
- `model.onnx`

Current export/profile status:

- ONNX export: `PASS`
- Profile updated for current `best.pt`: `PASS`

## Health Check

The long run is currently healthy:

- training resumed correctly
- losses continued updating normally
- periodic validation improved steadily from step `500` through step `1200`
- `best.pt` and `latest.pt` were updated correctly
- benchmark evaluation and export were refreshed for the new best checkpoint
- periodic step checkpoints now retain the latest numeric steps correctly

## Exact Command To Continue Later

```bash
PYTHONPATH=src .venv/bin/python scripts/run_local_full_x4.py
```

This command will continue the same real-data x4 mainline run from:

```text
outputs/run_x4_edge_default/latest.pt
```

## Honest Limitation

The configured full schedule of `100000` steps was not completed in this session. The run is still in progress conceptually, but it remains fully resumable from the saved step `1200` checkpoint.
