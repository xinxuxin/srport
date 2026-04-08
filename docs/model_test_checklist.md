# Model Test Checklist

## Static Checks

```bash
PYTHONPATH=src .venv/bin/ruff check src tests
PYTHONPATH=src .venv/bin/mypy src
```

## Full Model Test Suite

```bash
PYTHONPATH=src .venv/bin/pytest -q
```

## Focused Model Tests

```bash
PYTHONPATH=src .venv/bin/pytest tests/test_model.py -q
PYTHONPATH=src .venv/bin/pytest tests/test_model_blocks.py -q
PYTHONPATH=src .venv/bin/pytest tests/test_train.py -q
PYTHONPATH=src .venv/bin/pytest tests/test_metrics.py -q
PYTHONPATH=src .venv/bin/pytest tests/test_profiling.py -q
```

## CPU Training Smoke

```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
from pathlib import Path
from epnet.config import TrainConfig, model_config_from_variant
from epnet.train import train

train(
    train_dir=None,
    output_path=Path("outputs/check_cpu.pt"),
    model_config=model_config_from_variant("tiny", upscale=2),
    train_config=TrainConfig(
        scale=2,
        batch_size=2,
        total_steps=4,
        save_every=2,
        log_every=1,
        device="cpu",
    ),
    synthetic_count=64,
)
PY
```

## MPS Training Smoke

```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
from pathlib import Path
from epnet.config import TrainConfig, model_config_from_variant
from epnet.train import train

train(
    train_dir=None,
    output_path=Path("outputs/check_mps.pt"),
    model_config=model_config_from_variant("tiny", upscale=2),
    train_config=TrainConfig(
        scale=2,
        batch_size=2,
        total_steps=4,
        save_every=2,
        log_every=1,
        device="mps",
        amp="off",
    ),
    synthetic_count=64,
)
PY
```

## MPS Variant Comparison And Ablation

```bash
PYTHONPATH=src .venv/bin/python -m epnet.experiments \
  --output-dir outputs/model_ablation_mps \
  --report-path docs/model_ablation_results.md \
  --json-path docs/model_ablation_results.json \
  --device mps \
  --scale 2 \
  --steps 8 \
  --synthetic-count 128
```

## Inference Checkpoint Smoke

```bash
PYTHONPATH=src .venv/bin/epnet-infer \
  --checkpoint outputs/check_cpu_inference.pt \
  --input data/samples/demo_input.png \
  --output outputs/check_cpu_output.png \
  --device cpu
```

## Evaluation Smoke

```bash
mkdir -p outputs/check_eval
cp data/samples/demo_input.png outputs/check_eval/demo_input.png
PYTHONPATH=src .venv/bin/epnet-eval \
  --checkpoint outputs/check_cpu_inference.pt \
  --hr-dir outputs/check_eval \
  --device cpu
```
