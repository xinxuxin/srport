# Code Reading Guide

This guide tells you how to read the repository in a presentation-friendly
order. The goal is not to read every file linearly, but to build a mental model
of the system.

## Recommended Reading Order

### 1. Start with the architecture

Read these first:

1. [/Users/macbook/Desktop/epnet/src/epnet/models/epnet.py](/Users/macbook/Desktop/epnet/src/epnet/models/epnet.py)
2. [/Users/macbook/Desktop/epnet/src/epnet/models/blocks/pfem.py](/Users/macbook/Desktop/epnet/src/epnet/models/blocks/pfem.py)
3. [/Users/macbook/Desktop/epnet/src/epnet/models/blocks/espm.py](/Users/macbook/Desktop/epnet/src/epnet/models/blocks/espm.py)
4. [/Users/macbook/Desktop/epnet/src/epnet/models/blocks/dcab.py](/Users/macbook/Desktop/epnet/src/epnet/models/blocks/dcab.py)
5. [/Users/macbook/Desktop/epnet/src/epnet/models/blocks/global_context.py](/Users/macbook/Desktop/epnet/src/epnet/models/blocks/global_context.py)

Why:

- `epnet.py` tells you the full branch structure.
- `pfem.py` and `espm.py` explain the two main branches.
- `dcab.py` and `global_context.py` explain the most specific internal logic.

### 2. Then read how configs build the model

Read:

1. [/Users/macbook/Desktop/epnet/src/epnet/config.py](/Users/macbook/Desktop/epnet/src/epnet/config.py)
2. [/Users/macbook/Desktop/epnet/src/epnet/models/presets.py](/Users/macbook/Desktop/epnet/src/epnet/models/presets.py)
3. [/Users/macbook/Desktop/epnet/src/epnet/models/registry.py](/Users/macbook/Desktop/epnet/src/epnet/models/registry.py)

Why:

- These files explain how `paper_like`, `edge_tiny`, `edge_default`, and
  `balanced_quality` become concrete architectures.

### 3. Then read the training path

Read:

1. [/Users/macbook/Desktop/epnet/src/epnet/train.py](/Users/macbook/Desktop/epnet/src/epnet/train.py)
2. [/Users/macbook/Desktop/epnet/src/epnet/data/datamodule.py](/Users/macbook/Desktop/epnet/src/epnet/data/datamodule.py)
3. [/Users/macbook/Desktop/epnet/src/epnet/data/paired_dataset.py](/Users/macbook/Desktop/epnet/src/epnet/data/paired_dataset.py)
4. [/Users/macbook/Desktop/epnet/src/epnet/utils/metrics.py](/Users/macbook/Desktop/epnet/src/epnet/utils/metrics.py)

Why:

- This is the shortest path to understanding how one real training step works.

### 4. Then read evaluation and export

Read:

1. [/Users/macbook/Desktop/epnet/src/epnet/evaluate.py](/Users/macbook/Desktop/epnet/src/epnet/evaluate.py)
2. [/Users/macbook/Desktop/epnet/src/epnet/profile.py](/Users/macbook/Desktop/epnet/src/epnet/profile.py)
3. [/Users/macbook/Desktop/epnet/src/epnet/export.py](/Users/macbook/Desktop/epnet/src/epnet/export.py)

Why:

- These files show how a trained model is turned into benchmark metrics and
  deployment artifacts.

### 5. Then read deployment

Read:

1. [/Users/macbook/Desktop/epnet/src/epnet_api/app/core/settings.py](/Users/macbook/Desktop/epnet/src/epnet_api/app/core/settings.py)
2. [/Users/macbook/Desktop/epnet/src/epnet_api/app/services/inference.py](/Users/macbook/Desktop/epnet/src/epnet_api/app/services/inference.py)
3. [/Users/macbook/Desktop/epnet/src/epnet_api/app/api/routes.py](/Users/macbook/Desktop/epnet/src/epnet_api/app/api/routes.py)
4. [/Users/macbook/Desktop/epnet/src/epnet_api/app/main.py](/Users/macbook/Desktop/epnet/src/epnet_api/app/main.py)

Why:

- These files explain how the trained checkpoint becomes a user-facing product.

## How to Trace One Inference Request

Start from the API:

1. The frontend or CLI sends an image to the inference endpoint.
2. [/Users/macbook/Desktop/epnet/src/epnet_api/app/api/routes.py](/Users/macbook/Desktop/epnet/src/epnet_api/app/api/routes.py)
   validates the upload and passes bytes into the service layer.
3. [/Users/macbook/Desktop/epnet/src/epnet_api/app/services/inference.py](/Users/macbook/Desktop/epnet/src/epnet_api/app/services/inference.py)
   decodes the image, resolves the correct checkpoint/runtime, runs EPNet,
   stores output artifacts, and logs analytics.
4. The EPNet forward path lives in
   [/Users/macbook/Desktop/epnet/src/epnet/models/epnet.py](/Users/macbook/Desktop/epnet/src/epnet/models/epnet.py).
5. The response returns image URLs, runtime metrics, and pipeline timing.

## How to Trace One Training Run

1. A model config is loaded from `configs/model`.
2. A data config is loaded from `configs/data`.
3. A train config is loaded from `configs/train`.
4. [/Users/macbook/Desktop/epnet/src/epnet/train.py](/Users/macbook/Desktop/epnet/src/epnet/train.py)
   builds the model, optimizer, loaders, and EMA state.
5. [/Users/macbook/Desktop/epnet/src/epnet/data/datamodule.py](/Users/macbook/Desktop/epnet/src/epnet/data/datamodule.py)
   chooses real DIV2K data or synthetic smoke data.
6. The training loop writes:
   - `latest.pt`
   - `best.pt`
   - periodic `stepXXXX.pt`
   - `train_log.jsonl`
   - `manifest.json`
7. After training, evaluation/profile/export paths produce:
   - `eval.json`
   - `eval.md`
   - `profile.json`
   - `profile.md`
   - `model.onnx`

## What to Focus on for a Presentation

If you only have time to show a few files:

Model:

- [/Users/macbook/Desktop/epnet/src/epnet/models/epnet.py](/Users/macbook/Desktop/epnet/src/epnet/models/epnet.py)
- [/Users/macbook/Desktop/epnet/src/epnet/models/blocks/pfem.py](/Users/macbook/Desktop/epnet/src/epnet/models/blocks/pfem.py)
- [/Users/macbook/Desktop/epnet/src/epnet/models/blocks/espm.py](/Users/macbook/Desktop/epnet/src/epnet/models/blocks/espm.py)

Training:

- [/Users/macbook/Desktop/epnet/src/epnet/train.py](/Users/macbook/Desktop/epnet/src/epnet/train.py)

Deployment:

- [/Users/macbook/Desktop/epnet/src/epnet_api/app/services/inference.py](/Users/macbook/Desktop/epnet/src/epnet_api/app/services/inference.py)

## Shortcut Summary

- Start with `epnet.py` if you want to understand the architecture.
- Start with `train.py` if you want to understand the research pipeline.
- Start with `inference.py` if you want to understand the deployed product path.
