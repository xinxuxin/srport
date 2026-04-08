# Real Training Workflow

This repository now treats real-data training as the main research and local edge-training path.

## Main Flow

1. Download or place datasets under `data/raw/` and `data/benchmarks/`
2. Optionally precompute bicubic LR caches under `data/processed/`
3. Launch config-driven training with:
   - `python -m epnet.train --model-config ... --data-config ... --train-config ...`
4. Resume automatically from `outputs/<run_name>/latest.pt` when `auto_resume` is enabled
5. Evaluate the exported inference checkpoint across configured benchmarks
6. Generate `eval.json`, `eval.md`, `profile.json`, `profile.md`, and `model.onnx`

## Recommended Local Runs

- x4 real-data mainline:
  - `PYTHONPATH=src .venv/bin/python scripts/run_local_full_x4.py`
- x2 real-data local comparison:
  - `PYTHONPATH=src .venv/bin/python scripts/run_local_full_x2.py`
- synthetic regression smoke:
  - `PYTHONPATH=src .venv/bin/python -m epnet.train --model-config configs/model/edge_default.yaml --data-config configs/data/synthetic_x2.yaml --train-config configs/train/smoke.yaml`

## Run Directory Layout

Example:

```text
outputs/run_x4_edge_default/
  config_snapshot/
    data.json
    model.json
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

## Notes

- Synthetic runs are preserved for smoke/regression only.
- Real-data runs are intended to be resumable, reportable, and ready for local edge-oriented iteration.
- The implementation remains a paper-grounded approximation where the EPNet paper is underspecified.
