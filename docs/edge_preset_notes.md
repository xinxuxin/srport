# Edge Preset Notes

## Preset Roles

- `paper_like`
  - closest to the repository's paper-grounded default
  - used as a baseline comparison point
- `edge_tiny`
  - smallest preset for constrained demos and quick experimentation
- `edge_default`
  - recommended mainline local preset
  - combines lower PFEM depth and lower ESPM depth while preserving the EPNet design pattern
- `balanced_quality`
  - larger preset for local quality-oriented exploration

## Why `edge_default` Is Now the Mainline

The repository's synthetic and smoke-style local experiments suggested that the fully paper-like configuration is not the strongest default choice for local edge-oriented iteration.

The new `edge_default` uses:

- PFEM depth `n=2`
- ESPM levels `=2`
- the same overall EPNet structure:
  - shallow feature extraction
  - PFEM branch
  - ESPM branch
  - fusion by summation
  - reconstruction by convolution + PixelShuffle

## Global Context Options

PFEM now supports pluggable global context modes:

- `swin`
- `lightweight_conv_context`
- `none`

The default remains `swin`, because it is the closest to the current EPNet implementation already validated in this repo.

## Limitation

These presets are engineering presets for this repository. They should not be described as official EPNet variants from the paper.
