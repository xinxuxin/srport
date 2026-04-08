# Model Audit Report

## Scope

This audit covered only the EPNet model implementation and closely related model-side code:

- [src/epnet/model.py](/Users/macbook/Desktop/epnet/src/epnet/model.py)
- [src/epnet/modules.py](/Users/macbook/Desktop/epnet/src/epnet/modules.py)
- [src/epnet/config.py](/Users/macbook/Desktop/epnet/src/epnet/config.py)
- [src/epnet/train.py](/Users/macbook/Desktop/epnet/src/epnet/train.py)
- [src/epnet/evaluate.py](/Users/macbook/Desktop/epnet/src/epnet/evaluate.py)
- [src/epnet/infer.py](/Users/macbook/Desktop/epnet/src/epnet/infer.py)
- [src/epnet/profiling.py](/Users/macbook/Desktop/epnet/src/epnet/profiling.py)
- [src/epnet/metrics.py](/Users/macbook/Desktop/epnet/src/epnet/metrics.py)
- model-side tests under [tests](/Users/macbook/Desktop/epnet/tests)

## What Was Inspected

- top-level EPNet structure
- PFEM / ESPM / LFEB / ESAB / DCAB / transformer blocks
- x2 / x3 / x4 output scaling
- checkpoint save/load/export paths
- training loop, EMA, validation, and evaluation metrics
- profiling path for parameters, MACs, and FLOPs
- CPU / MPS runtime behavior

## Structure Found

Paper-derived design that is present:

- shallow `3x3` feature extraction convolution
- PFEM branch
- ESPM branch
- feature fusion by elementwise addition
- reconstruction through `3x3` convolution + `PixelShuffle`
- PFEM submodule composed of `LFEB + modified Swin transformer + ESAB`
- ESPM built around dynamic channel attention blocks and pyramid fusion

Implementation assumptions that remain approximate:

- exact Swin substructure and hyperparameters are an engineering approximation
- ESAB and ESPM internals are faithful approximations, not code-level paper-exact recovery
- profiler MACs/FLOPs remain engineering estimates, not a paper-original Multi-Adds reproduction

## Issues Found Initially

1. `profile_model()` silently over-counted MACs and FLOPs by multiplying them with the number of profiling iterations.
2. `ModelConfig` accepted invalid values such as `upscale=1`, `num_pfem=0`, `espm_levels=1`, invalid `split_ratio`, and incompatible `embed_dim/num_heads`.
3. `TrainConfig` lacked validation for clearly invalid values.
4. `evaluate_prediction()` could silently crop away the full image when `shave` was too large.
5. Missing checkpoint paths failed indirectly through `torch.load()` rather than with a clearer model-side message.
6. Test coverage was missing module-level forward checks and profiling-specific checks.

## Fixes Applied

- fixed profiling so MACs/FLOPs are counted from a single representative forward pass while latency is averaged separately
- added `ModelConfig.__post_init__()` validation for scale, PFEM depth, ESPM levels, split ratio, and head/channel compatibility
- added `TrainConfig.__post_init__()` validation for training hyperparameters and runtime settings
- added guardrails for excessive border shave in metric evaluation
- improved checkpoint loading error clarity with explicit missing-file checks
- added module-level tests for:
  - LFEB
  - ESAB
  - DCAB
  - ESPM
  - PFEM
  - SwinBlock
- added config validation tests
- added validation smoke tests
- added profiling stability tests
- added missing-checkpoint-path test

## Runtime Verification Performed

- default EPNet forward pass
- x2 / x3 / x4 output shape checks
- no-NaN forward checks
- module-level forward smoke tests
- checkpoint save/load/export tests
- tiny training smoke tests
- validation smoke tests
- profiling smoke tests
- CPU inference checkpoint smoke test
- local MPS training and inference smoke tests

## Local MPS Training Result

Local MPS runs were completed successfully using synthetic training data and a single-image validation folder:

- `tiny x2`, 8 steps:
  - loss decreased from `0.732571` to `0.364223`
  - validation PSNR improved from `7.0296` to `7.0440`
  - validation SSIM improved from `0.10988` to `0.11037`
  - exported [outputs/mps_tiny_x2_inference.pt](/Users/macbook/Desktop/epnet/outputs/mps_tiny_x2_inference.pt)
- `paper x2`, 8 steps:
  - loss decreased from `0.803760` to `0.384703`
  - validation PSNR improved from `5.9055` to `5.9204`
  - validation SSIM improved from `0.03019` to `0.03026`
  - exported [outputs/mps_paper_x2_inference.pt](/Users/macbook/Desktop/epnet/outputs/mps_paper_x2_inference.pt)

These runs confirm the MPS training path works. They are not quality benchmarks because they use synthetic data and a trivial validation setup.

## Remaining Assumptions / Approximations

- the architecture remains a faithful approximation of EPNet, not a code-perfect reconstruction from official source
- model-quality conclusions cannot be drawn from the local synthetic smoke runs
- CUDA support is implemented in the training/runtime path, but CUDA was not available on this machine for live verification
- profiler values are stable and consistent now, but they remain approximate engineering estimates
