# EPNet Reproduction Assumptions

## Paper-Derived Details

- The top-level EPNet pipeline follows the paper: shallow `3x3` convolution, PFEM branch, ESPM branch, feature fusion by element-wise addition, and image reconstruction with `3x3` convolution plus `PixelShuffle`.
- PFEM uses `n = 4` submodules by default.
- PFEM is built from LFEB, a modified Swin Transformer, and ESAB.
- ESPM uses dynamic channel separation through DCAB and a pyramid-style fusion strategy.
- Training defaults match the appendix: patch size `48`, batch size `32`, Adam with learning rate `5e-4` and betas `(0.9, 0.99)`, no weight decay, L1 loss, EMA decay `0.999`, `1e6` iterations, and no warm-up.
- Evaluation reports PSNR and SSIM.

## Engineering Assumptions

- The paper does not publish source code or enough tensor-level detail to uniquely recover the exact module wiring, channel sizes, or Swin hyperparameters. This implementation therefore uses a faithful approximation of the published diagrams and text.
- The default embedding dimension is set to `40` to keep the model close to the paper's reported `~485K` parameter regime while preserving a simple, readable implementation. This can be changed through configuration.
- PFEM submodules are not weight-shared by default. The paper text is ambiguous on the phrase "parameter-sharing LFEB and ESAB", so weight sharing is exposed as a configuration option instead of being forced.
- The modified Swin Transformer is implemented as two windowed self-attention blocks with alternating shifted windows, preserving the Swin-style local-global trade-off without introducing hierarchical patch merging.
- LFEB is implemented as `Conv3x3 -> GELU -> Conv3x3 -> ECAM + residual`, based on Figure 3 and the ECANet reference cited in the paper.
- ESAB is implemented as a spatial-attention residual block with pooled context, refinement convolutions, upsampling, sigmoid gating, and residual fusion, matching the intent and figure structure even though kernel sizes and exact channel widths are not fully specified in the paper text.
- ESPM is implemented as a three-level pyramid with DCAB at each level, bilinear top-down fusion, and channel-attention-guided fusion. The paper states the module is FPN-inspired but does not specify exact pooling or fusion operators.
- DCAB uses the published weighted combinatorial crossover equation and applies channel attention independently to the split tensors before recombination and fusion.
- The profiler reports estimated MACs and FLOPs from module hooks. These are approximate engineering estimates intended for local comparison, not a strict reproduction of the paper's exact Multi-Adds numbers.
- Evaluation uses standard lightweight SISR practice by computing PSNR and SSIM on the luminance (`Y`) channel after shaving border pixels equal to the upscale factor. The paper names the metrics and benchmark datasets but does not spell out the exact metric protocol.

## Optional Improvements

- A small bootstrap checkpoint can be trained on synthetic patterns for a working local demo before a full DIV2K run is available. This is a demo convenience and is not presented as a paper-level reproduction result.
- The FastAPI service records usage analytics in SQLite and exposes dashboard-ready summary endpoints. This is outside the scope of the paper and exists purely for the product demo.
- The web frontend emphasizes polished interaction, observability, and reproducibility rather than matching any paper figure.
