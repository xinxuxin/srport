# Design Notes for Presentation

This document is meant to help you explain the repository clearly to a strong
technical audience.

## Core Design Idea in Plain English

EPNet is treated here as a deployable image enhancement system, not only as a
paper reproduction. The architecture combines:

- a detail-focused path
- a lightweight pyramid/context path
- a reconstruction head that turns fused features into a high-resolution image

The codebase then extends that model with:

- real-data training
- resumable checkpoints
- evaluation and export
- deployment and serving
- a frontend/backend demo system

## Why the Architecture Is Split into Branches

PFEM branch:

- responsible for progressively refining features
- mixes local extraction, global context, and spatial attention
- best way to describe it verbally:
  "this is the rich detail-enrichment path"

ESPM branch:

- responsible for lower-cost multi-scale structure modeling
- best way to describe it verbally:
  "this is the efficient pyramid context path"

Reconstruction head:

- fuses both branches
- projects the fused representation into a PixelShuffle-ready tensor
- best way to describe it verbally:
  "this turns learned feature maps into the final super-resolved image"

## Why the Presets Exist

`paper_like`:

- baseline closest to the paper-driven structure
- useful when discussing fidelity to the publication

`edge_tiny`:

- smallest experimental/deployment-friendly variant

`edge_default`:

- main product preset
- used because the local engineering experiments showed a better practical
  trade-off than the paper-like preset under repository constraints

`balanced_quality`:

- wider comparison point for local quality-oriented experiments

## What Is Research-Oriented vs Deployment-Oriented

Research-oriented:

- `src/epnet/models`
- `src/epnet/train.py`
- `src/epnet/evaluate.py`
- `src/epnet/export.py`
- `src/epnet/profile.py`
- `configs/`

Deployment-oriented:

- `src/epnet_api`
- `frontend`
- `scripts/run_local.sh`

Shared infrastructure:

- `src/epnet/data`
- `src/epnet/utils`

## Key Technical Trade-Offs

Paper fidelity vs deployability:

- the repository keeps a `paper_like` baseline
- the default deployed model is `edge_default`

Dynamic deployment vs export simplicity:

- PyTorch is the most robust default inference backend
- ONNX export is supported, but deployment keeps PyTorch as the default because
  dynamic runtime robustness matters more than claiming universal exportability

Large-image support vs latency:

- tile inference is available for memory safety
- full-image inference is simpler and usually faster for small inputs

Synthetic testing vs real-data training:

- synthetic data stays in the repo as a smoke/regression path
- real DIV2K-based training is the mainline path

## Best Files to Show in a Presentation

If you want one architecture file:

- [/Users/macbook/Desktop/epnet/src/epnet/models/epnet.py](/Users/macbook/Desktop/epnet/src/epnet/models/epnet.py)

If you want one block-level file:

- [/Users/macbook/Desktop/epnet/src/epnet/models/blocks/pfem.py](/Users/macbook/Desktop/epnet/src/epnet/models/blocks/pfem.py)

If you want one training file:

- [/Users/macbook/Desktop/epnet/src/epnet/train.py](/Users/macbook/Desktop/epnet/src/epnet/train.py)

If you want one deployment file:

- [/Users/macbook/Desktop/epnet/src/epnet_api/app/services/inference.py](/Users/macbook/Desktop/epnet/src/epnet_api/app/services/inference.py)

## What to Explain Verbally

Most important verbal points:

1. why PFEM and ESPM exist as separate branches
2. why `edge_default` is the deployment preset instead of `paper_like`
3. how the training pipeline produces a deployment-specific inference artifact
4. how the API serves that artifact through a stable preprocessing/postprocessing path
5. how the frontend turns that into an understandable product demo

## Suggested Presentation Flow

1. Start with the problem: single-image super-resolution that is good enough to
   deploy, not just benchmark.
2. Show `epnet.py` and explain the two-branch architecture.
3. Show `train.py` and explain how the repository makes the model reproducible.
4. Show `inference.py` and explain how the trained checkpoint becomes a service.
5. Show the frontend/system demo as proof that the model is integrated end to end.
