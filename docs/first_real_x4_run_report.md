# First Real x4 Run Report

## Scope

This report covers the first successful local real-data EPNet x4 run after the research/edge-training refactor. The goal was not to finish the full 100k-step training schedule in one session, but to:

- make the real-data path pass on this machine
- start a genuine x4 DIV2K training run
- verify checkpointing and auto-resume
- generate first real evaluation artifacts
- make ONNX/export smoke pass

## Commands Run

```bash
PYTHONPATH=src .venv/bin/python scripts/download_real_data.py --data-root data
PYTHONPATH=src .venv/bin/python scripts/prepare_real_data.py --data-root data --scales 2 3 4
PYTHONPATH=src .venv/bin/python scripts/run_local_full_x4.py
PYTHONPATH=src .venv/bin/python -m epnet.evaluate \
  --checkpoint outputs/run_x4_edge_default/best.pt \
  --data-config configs/data/div2k_x4.yaml \
  --output-json outputs/run_x4_edge_default/eval.json \
  --output-markdown outputs/run_x4_edge_default/eval.md \
  --device mps
PYTHONPATH=src .venv/bin/python -m epnet.export \
  --checkpoint outputs/run_x4_edge_default/best.pt \
  --output outputs/run_x4_edge_default/model.onnx \
  --device cpu
PYTHONPATH=src .venv/bin/python - <<'PY'
from pathlib import Path
from epnet.profile import profile_checkpoint
print(
    profile_checkpoint(
        Path("outputs/run_x4_edge_default/best.pt"),
        device="cpu",
        onnx_export_path=Path("outputs/run_x4_edge_default/model.onnx"),
        output_json=Path("outputs/run_x4_edge_default/profile.json"),
        output_markdown=Path("outputs/run_x4_edge_default/profile.md"),
    )
)
PY
PYTHONPATH=src .venv/bin/ruff check src tests scripts
PYTHONPATH=src .venv/bin/mypy src
PYTHONPATH=src .venv/bin/pytest -q
```

## Datasets Actually Prepared

- `DIV2K_train_HR`: present, 800 HR images
- `DIV2K_valid_HR`: present, 100 HR images
- `Set5/HR`: present, 5 images
- `Set14/HR`: present, 14 images
- `BSD100/HR`: present, 100 images
- `Urban100/HR`: present, 100 images
- `Manga109/HR`: not prepared automatically by design; still manual because of its separate usage agreement

Prepared bicubic caches were also generated for `X2/X3/X4` under:

- `data/processed/DIV2K_train_LR_bicubic`
- `data/processed/DIV2K_valid_LR_bicubic`
- `data/processed/benchmarks/{Set5,Set14,BSD100,Urban100}/LR_bicubic`

## Real-Data Setup Status

- Real-data path setup: `PASS`

`validate_real_data_paths(Path("data"))` now succeeds, and the public benchmark datasets needed for the default evaluation path are available locally.

### Dataset source note

The official VDSR benchmark mirror (`https://cv.snu.ac.kr/research/VDSR/test_data.zip`) failed on this machine with `Connection reset by peer`. The repository download script now falls back to public Hugging Face dataset mirrors for:

- `Set5`
- `Set14`
- `BSD100`
- `Urban100`

Manga109 remains manual and is not required for the main x4 path.

## Real x4 Training Launch

- Training launch: `PASS`
- Device used: `mps`
- Model preset: `edge_default`
- Scale: `x4`
- Train dataset: `DIV2K`
- Output directory: `outputs/run_x4_edge_default`

The run was started through the intended real-data script:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_local_full_x4.py
```

This was a genuine real-data run using `data/raw/DIV2K/DIV2K_train_HR`, not the synthetic smoke path.

## Checkpointing and Resume

- Checkpoint generation: `PASS`
- Auto-resume: `PASS`

Artifacts generated during this session:

- `outputs/run_x4_edge_default/manifest.json`
- `outputs/run_x4_edge_default/train_log.jsonl`
- `outputs/run_x4_edge_default/latest.pt`
- `outputs/run_x4_edge_default/best.pt`
- `outputs/run_x4_edge_default/step200.pt`
- `outputs/run_x4_edge_default/step300.pt`
- `outputs/run_x4_edge_default/step400.pt`

Observed training progress:

- step `100`: validation PSNR `10.5642`, SSIM `0.03340`
- step `200`: validation PSNR `11.5542`, SSIM `0.04202`
- step `300`: validation PSNR `12.6397`, SSIM `0.05461`
- step `400`: validation PSNR `13.8445`, SSIM `0.07248`

Resume was verified by interrupting the run after checkpoint creation and relaunching `scripts/run_local_full_x4.py`. The resumed run logged from step `225`, confirming that it reloaded `outputs/run_x4_edge_default/latest.pt` from step `200` and continued rather than restarting from scratch.

At the end of this session:

- `latest.pt` step: `400`
- `best.pt` step: `400`

## Real Evaluation Artifacts

- Real-data evaluation artifact generation: `PASS`

Generated files:

- `outputs/run_x4_edge_default/eval.json`
- `outputs/run_x4_edge_default/eval.md`

Datasets actually evaluated:

- `Set5`
- `Set14`
- `BSD100`
- `Urban100`

Datasets not evaluated:

- `Manga109` was not available locally

Current early-run evaluation summary for `best.pt`:

- Samples: `219`
- Mean PSNR: `13.8477`
- Mean SSIM: `0.09724`

Per-dataset means:

- `Set5`: PSNR `13.8671`, SSIM `0.11495`
- `Set14`: PSNR `13.7445`, SSIM `0.09359`
- `BSD100`: PSNR `14.5200`, SSIM `0.09516`
- `Urban100`: PSNR `13.2591`, SSIM `0.08525`

These are first-run early-training artifacts only. They are useful for verifying the real pipeline and comparing future runs, but they are **not** final benchmark claims.

## Export / Profile

- Export smoke: `PASS`

Generated files:

- `outputs/run_x4_edge_default/model.onnx`
- `outputs/run_x4_edge_default/profile.json`
- `outputs/run_x4_edge_default/profile.md`

Profile summary:

- Parameters: `264,854`
- MACs: `578,604,160`
- FLOPs: `1,157,208,320`
- CPU latency: `50.37 ms`
- Checkpoint size: `4,655,303 bytes`
- Model state size: `1,190,488 bytes`
- Estimated memory footprint: `2,380,976 bytes`

Export note:

- ONNX export now succeeds after installing the missing `onnx` dependency.
- The export path still emits PyTorch trace warnings from the windowed global-context block; these are warnings, not a smoke-test failure.

## What Did Not Fully Complete

The full `100000`-step x4 run did **not** complete in this session. That is expected and was not faked. What was completed is:

- real-data setup
- real x4 training launch
- first real checkpoints
- auto-resume verification
- real benchmark evaluation artifacts
- ONNX export smoke

## Final Status

- Real-data path setup: `PASS`
- x4 real-data training launch: `PASS`
- checkpoint generation: `PASS`
- checkpoint resume: `PASS`
- real-data evaluation artifact generation: `PASS`
- export smoke: `PASS`
