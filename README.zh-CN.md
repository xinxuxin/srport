# EPNet Super-Resolution System

This repository now uses English-only documentation for code comments and study
guides so the presentation material stays consistent. This file remains in the
repository as a mirrored entrypoint for viewers who expect a second README in
the root.

Primary guide:

- [/Users/macbook/Desktop/epnet/README.md](/Users/macbook/Desktop/epnet/README.md)

## What This Repository Is

This project turns EPNet from a model implementation into a full system:

- model architecture
- real-data training
- evaluation
- export
- deployment
- frontend/backend demo integration

## Main Directories

- `src/epnet/models`
  - architecture and block-level implementation
- `src/epnet/data`
  - synthetic and real-data pipelines
- `src/epnet/train.py`
  - training entrypoint
- `src/epnet/evaluate.py`
  - evaluation entrypoint
- `src/epnet/export.py`
  - ONNX export path
- `src/epnet_api`
  - FastAPI deployment system
- `frontend`
  - interactive demo UI
- `configs`
  - model, data, and train configuration
- `docs`
  - study guides and system reports

## Recommended Reading Order

Model understanding:

1. [/Users/macbook/Desktop/epnet/src/epnet/models/epnet.py](/Users/macbook/Desktop/epnet/src/epnet/models/epnet.py)
2. [/Users/macbook/Desktop/epnet/src/epnet/models/blocks/pfem.py](/Users/macbook/Desktop/epnet/src/epnet/models/blocks/pfem.py)
3. [/Users/macbook/Desktop/epnet/src/epnet/models/blocks/espm.py](/Users/macbook/Desktop/epnet/src/epnet/models/blocks/espm.py)

Training understanding:

1. [/Users/macbook/Desktop/epnet/src/epnet/train.py](/Users/macbook/Desktop/epnet/src/epnet/train.py)
2. [/Users/macbook/Desktop/epnet/src/epnet/data/datamodule.py](/Users/macbook/Desktop/epnet/src/epnet/data/datamodule.py)
3. [/Users/macbook/Desktop/epnet/src/epnet/data/paired_dataset.py](/Users/macbook/Desktop/epnet/src/epnet/data/paired_dataset.py)

Deployment understanding:

1. [/Users/macbook/Desktop/epnet/src/epnet_api/app/services/inference.py](/Users/macbook/Desktop/epnet/src/epnet_api/app/services/inference.py)
2. [/Users/macbook/Desktop/epnet/src/epnet_api/app/api/routes.py](/Users/macbook/Desktop/epnet/src/epnet_api/app/api/routes.py)
3. [/Users/macbook/Desktop/epnet/src/epnet_api/app/core/settings.py](/Users/macbook/Desktop/epnet/src/epnet_api/app/core/settings.py)

## Main Workflows

### Synthetic smoke path

```bash
PYTHONPATH=src .venv/bin/python -m epnet.train \
  --model-config configs/model/edge_default_x2.yaml \
  --data-config configs/data/synthetic_x2.yaml \
  --train-config configs/train/smoke.yaml
```

### Real-data full x4 path

```bash
PYTHONPATH=src .venv/bin/python scripts/download_real_data.py --data-root data
PYTHONPATH=src .venv/bin/python scripts/prepare_real_data.py --data-root data --scales 2 3 4
PYTHONPATH=src .venv/bin/python scripts/run_local_full_x4.py
```

### Evaluation

```bash
PYTHONPATH=src .venv/bin/python -m epnet.evaluate \
  --checkpoint outputs/run_x4_edge_default/inference.pt \
  --data-config configs/data/div2k_x4.yaml \
  --output-json outputs/run_x4_edge_default/eval.json \
  --output-markdown outputs/run_x4_edge_default/eval.md
```

### Deployment demo

```bash
./scripts/run_local.sh
```

## Important Supporting Docs

- paper-to-code mapping:
  [/Users/macbook/Desktop/epnet/docs/paper_to_code_map.md](/Users/macbook/Desktop/epnet/docs/paper_to_code_map.md)
- config reference:
  [/Users/macbook/Desktop/epnet/docs/config_reference.md](/Users/macbook/Desktop/epnet/docs/config_reference.md)
- code reading guide:
  [/Users/macbook/Desktop/epnet/docs/code_reading_guide.md](/Users/macbook/Desktop/epnet/docs/code_reading_guide.md)
- design notes for presentation:
  [/Users/macbook/Desktop/epnet/docs/design_notes_for_presentation.md](/Users/macbook/Desktop/epnet/docs/design_notes_for_presentation.md)
- dataset setup:
  [/Users/macbook/Desktop/epnet/docs/dataset_setup.md](/Users/macbook/Desktop/epnet/docs/dataset_setup.md)
- real training workflow:
  [/Users/macbook/Desktop/epnet/docs/real_training_workflow.md](/Users/macbook/Desktop/epnet/docs/real_training_workflow.md)

## Deployment Defaults

Default deployed artifact:

- `/Users/macbook/Desktop/epnet/outputs/run_x4_edge_default/inference.pt`

Default backend:

- PyTorch

Optional backend:

- ONNX when the environment supports it

## Notes

- The repository is a documented, engineering-grounded implementation of EPNet,
  not an official author code release.
- Synthetic experiments are regression aids, not benchmark claims.
- Manga109 remains a manual dataset path.
