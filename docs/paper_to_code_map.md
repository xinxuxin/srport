# Paper-to-Code Map

This document explains how the EPNet paper concepts map onto the implementation
in this repository. It is intended as a study guide for presentations, code
reviews, and onboarding rather than as a claim of official paper-source parity.

## Reading This Map

Use this file together with:

- [/Users/macbook/Desktop/epnet/docs/code_reading_guide.md](/Users/macbook/Desktop/epnet/docs/code_reading_guide.md)
- [/Users/macbook/Desktop/epnet/docs/config_reference.md](/Users/macbook/Desktop/epnet/docs/config_reference.md)
- [/Users/macbook/Desktop/epnet/docs/epnet_assumptions.md](/Users/macbook/Desktop/epnet/docs/epnet_assumptions.md)

## Top-Level Architecture

The paper describes EPNet as a combination of:

1. a shallow feature extraction stem
2. a PFEM branch
3. an ESPM branch
4. feature fusion
5. a reconstruction head using convolution plus PixelShuffle

Code mapping:

- Top-level EPNet model:
  [/Users/macbook/Desktop/epnet/src/epnet/models/epnet.py](/Users/macbook/Desktop/epnet/src/epnet/models/epnet.py)
- Presets and recommended deployment variants:
  [/Users/macbook/Desktop/epnet/src/epnet/models/presets.py](/Users/macbook/Desktop/epnet/src/epnet/models/presets.py)
- Model construction entrypoints:
  [/Users/macbook/Desktop/epnet/src/epnet/models/registry.py](/Users/macbook/Desktop/epnet/src/epnet/models/registry.py)

## PFEM Mapping

Paper concept:

- PFEM is the panoramic feature extraction branch.
- Each PFEM stage combines local feature extraction, transformer-style context
  modeling, and spatial attention refinement.

Code mapping:

- PFEM stage wrapper:
  [/Users/macbook/Desktop/epnet/src/epnet/models/blocks/pfem.py](/Users/macbook/Desktop/epnet/src/epnet/models/blocks/pfem.py)
- LFEB:
  [/Users/macbook/Desktop/epnet/src/epnet/models/blocks/lfeb.py](/Users/macbook/Desktop/epnet/src/epnet/models/blocks/lfeb.py)
- Global context block:
  [/Users/macbook/Desktop/epnet/src/epnet/models/blocks/global_context.py](/Users/macbook/Desktop/epnet/src/epnet/models/blocks/global_context.py)
- ESAB:
  [/Users/macbook/Desktop/epnet/src/epnet/models/blocks/esab.py](/Users/macbook/Desktop/epnet/src/epnet/models/blocks/esab.py)

Implementation notes:

- The repository uses a pluggable global-context abstraction:
  - `swin`
  - `lightweight_conv_context`
  - `none`
- The default deployment-oriented preset still uses the Swin-like path because
  it best matches the paper's intent.
- The modular abstraction is an engineering improvement that makes future edge
  simplification easier without rewriting PFEM.

## ESPM Mapping

Paper concept:

- ESPM is the efficient spatial pyramid branch.
- It provides broader multi-scale structure and edge-aware context.

Code mapping:

- ESPM:
  [/Users/macbook/Desktop/epnet/src/epnet/models/blocks/espm.py](/Users/macbook/Desktop/epnet/src/epnet/models/blocks/espm.py)
- DCAB:
  [/Users/macbook/Desktop/epnet/src/epnet/models/blocks/dcab.py](/Users/macbook/Desktop/epnet/src/epnet/models/blocks/dcab.py)

Implementation notes:

- The paper figure shows a split, transform, channel-attention, concat, and
  fusion pattern. The code follows that idea directly in DCAB.
- The pyramid depth is configurable through `espm_levels`.
- The deployment-oriented `edge_default` preset reduces ESPM depth from the
  paper-like baseline to improve efficiency.

## Reconstruction Head Mapping

Paper concept:

- PFEM and ESPM features are fused.
- A convolution projects the fused features into a PixelShuffle-ready tensor.
- PixelShuffle reconstructs the super-resolved image.

Code mapping:

- Feature fusion and reconstruction head:
  [/Users/macbook/Desktop/epnet/src/epnet/models/epnet.py](/Users/macbook/Desktop/epnet/src/epnet/models/epnet.py)

Implementation notes:

- The repository keeps this part intentionally straightforward because it is one
  of the clearest and most paper-faithful sections of the network.

## Presets: Research Baseline vs Edge Variants

Research-oriented baseline:

- `paper_like`
  - intended for comparison against the paper-style structure
  - not the default product deployment path

Edge-oriented presets:

- `edge_tiny`
  - smallest preset
  - useful for smoke tests and lightweight deployment experiments
- `edge_default`
  - default product/demo preset
  - PFEM depth `2`
  - ESPM levels `2`
  - chosen because local engineering experiments showed a better quality/latency
    trade-off under practical constraints
- `balanced_quality`
  - wider model for local comparison when quality is prioritized over speed

Preset definitions:

- [/Users/macbook/Desktop/epnet/src/epnet/models/presets.py](/Users/macbook/Desktop/epnet/src/epnet/models/presets.py)
- [/Users/macbook/Desktop/epnet/src/epnet/config.py](/Users/macbook/Desktop/epnet/src/epnet/config.py)

## Training, Evaluation, and Deployment Additions

The paper focuses on model design and benchmarking, while the repository adds
system-level capabilities required for a real product/demo.

Research pipeline:

- Training:
  [/Users/macbook/Desktop/epnet/src/epnet/train.py](/Users/macbook/Desktop/epnet/src/epnet/train.py)
- Evaluation:
  [/Users/macbook/Desktop/epnet/src/epnet/evaluate.py](/Users/macbook/Desktop/epnet/src/epnet/evaluate.py)
- Profiling:
  [/Users/macbook/Desktop/epnet/src/epnet/profile.py](/Users/macbook/Desktop/epnet/src/epnet/profile.py)
- Export:
  [/Users/macbook/Desktop/epnet/src/epnet/export.py](/Users/macbook/Desktop/epnet/src/epnet/export.py)

Deployment additions:

- FastAPI app assembly:
  [/Users/macbook/Desktop/epnet/src/epnet_api/app/main.py](/Users/macbook/Desktop/epnet/src/epnet_api/app/main.py)
- Inference service:
  [/Users/macbook/Desktop/epnet/src/epnet_api/app/services/inference.py](/Users/macbook/Desktop/epnet/src/epnet_api/app/services/inference.py)
- API routes:
  [/Users/macbook/Desktop/epnet/src/epnet_api/app/api/routes.py](/Users/macbook/Desktop/epnet/src/epnet_api/app/api/routes.py)
- Analytics:
  [/Users/macbook/Desktop/epnet/src/epnet_api/app/services/analytics.py](/Users/macbook/Desktop/epnet/src/epnet_api/app/services/analytics.py)

These deployment modules are not paper modules. They are product-oriented
infrastructure that turns a trained checkpoint into a repeatable demo system.

## Where the Repository Follows the Paper Closely

- shallow `3x3` stem
- PFEM branch as a repeated sequence of LFEB + context block + ESAB
- ESPM branch with DCAB-inspired split/fuse behavior
- PFEM + ESPM fusion
- reconstruction through convolution + PixelShuffle
- support for x2, x3, and x4
- paper-style optimizer defaults and loss choices in the training configs

## Where the Repository Intentionally Differs

- preset system with `edge_default` as the mainline demo model
- configurable global-context backend inside PFEM
- config-driven training/evaluation/export workflow
- synthetic smoke path for CI and fast regression checks
- ONNX export and deployment artifacts
- analytics, replay history, and frontend/backend integration

## Where to Look If You Need to Explain the Architecture Quickly

1. [/Users/macbook/Desktop/epnet/src/epnet/models/epnet.py](/Users/macbook/Desktop/epnet/src/epnet/models/epnet.py)
2. [/Users/macbook/Desktop/epnet/src/epnet/models/blocks/pfem.py](/Users/macbook/Desktop/epnet/src/epnet/models/blocks/pfem.py)
3. [/Users/macbook/Desktop/epnet/src/epnet/models/blocks/espm.py](/Users/macbook/Desktop/epnet/src/epnet/models/blocks/espm.py)
4. [/Users/macbook/Desktop/epnet/src/epnet/models/blocks/dcab.py](/Users/macbook/Desktop/epnet/src/epnet/models/blocks/dcab.py)
5. [/Users/macbook/Desktop/epnet/src/epnet_api/app/services/inference.py](/Users/macbook/Desktop/epnet/src/epnet_api/app/services/inference.py)
