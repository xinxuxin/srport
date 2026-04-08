# Model Optimization Report

## Implemented Optimizations

### Profiling correctness and stability

- `profile_model()` now counts MACs/FLOPs from one representative forward pass instead of accumulating them across timing iterations.
- latency measurement still averages across multiple iterations, so the timing path remains smooth without corrupting MAC totals.

### Safer runtime behavior

- evaluation and inference already used `torch.inference_mode()` and were preserved that way
- training/device flow now centralizes runtime choice in [device.py](/Users/macbook/Desktop/epnet/src/epnet/device.py)
- channels-last layout is used only where it is useful and supported

### Checkpoint lifecycle

- training exports a dedicated `*_inference.pt` checkpoint using EMA weights
- missing checkpoint paths now fail earlier and more clearly

### Config reproducibility

- model and training configs now validate invalid settings early instead of failing later inside unrelated modules

## Future Low-Risk Optimizations

- cache profiling results per `(variant, scale, input_shape)` when repeated comparisons are needed
- optionally support bfloat16 autocast on future hardware/runtime combinations once verified stable
- add deterministic-mode toggle for strict reproducibility experiments
- add a lightweight benchmark helper to compare `tiny / paper / balanced` on identical synthetic or real mini-val subsets
- consider optional fused or reduced-permutation attention implementations only if clarity is preserved

## Deliberately Not Optimized

- PFEM / ESPM internal structure was not rewritten into a different architecture
- no speculative quantization, pruning, or architecture surgery was introduced
- no attempt was made to overfit synthetic smoke runs into misleading “quality improvements”
