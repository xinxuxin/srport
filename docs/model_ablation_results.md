# EPNet Ablation Results

## Experiment Setup

- Device: `mps`
- Scale: `x2`
- Steps per run: `8`
- Synthetic training samples: `128`
- Validation folder: `outputs/model_ablation_mps/synthetic_eval`
- Validation data: deterministic synthetic images generated locally for reproducible smoke-style comparison
- Important note: these are engineering comparison runs, not paper benchmark claims

## Variant Comparison

| Name | Variant | Params | MACs | Latency (ms) | Avg PSNR | Avg SSIM | Best PSNR | Train Time (s) | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| variant_tiny | tiny | 245,053 | 493,366,784 | 9.93 | 8.8270 | 0.04956 | 8.8270 | 3.81 | Tiny preset for minimum parameter budget. |
| variant_balanced | balanced | 647,160 | 1,317,416,832 | 12.70 | 7.6564 | 0.05919 | 7.6564 | 4.00 | Slightly wider model for capacity/performance tradeoff. |
| variant_paper | paper | 450,864 | 933,903,840 | 13.88 | 7.4922 | 0.04047 | 7.4922 | 7.20 | Paper-grounded default preset. |

## Ablation Comparison

| Name | Variant | Params | MACs | Latency (ms) | Avg PSNR | Avg SSIM | Best PSNR | Train Time (s) | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ablation_espm2 | paper | 398,584 | 919,155,840 | 11.50 | 9.1427 | 0.05067 | 9.1427 | 1.72 | Reduce ESPM pyramid depth from 3 to 2. |
| ablation_pfem2 | paper | 304,138 | 563,492,320 | 9.21 | 9.1054 | 0.05821 | 9.1054 | 1.27 | Reduce PFEM depth from n=4 to n=2. |
| ablation_shared_pfem | paper | 230,775 | 933,903,840 | 12.15 | 8.6270 | 0.08028 | 8.6270 | 1.13 | Share PFEM block weights across repeated applications. |

## Key Takeaways

- Best local PSNR in this run set: `ablation_espm2` with `9.1427` dB.
- Fastest profiled checkpoint: `ablation_pfem2` at `9.21` ms.
- Smallest model: `ablation_shared_pfem` with `230,775` parameters.
- Interpretation warning: because the training and validation data are synthetic and very small, these results are only appropriate for relative smoke-style comparisons on this machine.
