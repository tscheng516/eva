# BACH Evaluation Speedup Summary

This document summarizes the top bottlenecks found during BACH offline evaluation, with ranked impact, root causes, applied fixes, and measured wall-clock results.

## 1) Slow train step in fit stage

### Why it was slow
Embeddings were repeatedly shipped and loaded for each epoch, creating avoidable host-side overhead.

### Fix
- Added cache-in-memory in predict stage to keep embeddings in RAM for downstream fit/validate use.
- Added preload behavior so fit/validate can still benefit when predict is skipped.

### Result
Wall-clock improved from 2m23s to 1m8s.

## 2) Slow preprocessing and CPU-bound pipeline

### Why it was slow
Profiling showed CPU time higher than CUDA time, which is common for image workloads with relatively small models and large input images.

### Fix A
- Added Triton kernel path to fuse rescale + normalize on GPU instead of CPU.

### Result A
Minimal speedup.

### Future direction for Fix A
Extend fusion to include resize + crop with sharded image data, where GPU tensor execution can provide more benefit.

### Fix B
- Reduced predict_batch_size from 64 to 2 to reduce CPU pressure.

### Result B
Wall-clock improved from 1m7s to 1m4s.

## 3) Evaluation at every training epoch (redundant and CPU-heavy)

### Why it was slow
Validation frequency was too high relative to the short max_steps run, causing repeated expensive eval overhead.

### Fix
- Implemented step-based validation control via val_check_interval.
- Added logic to skip final post-fit validation when max_steps is divisible by validation interval and fit already validated at the terminal step.

### Result
Wall-clock improved from 1m4s to 51s.

## Final Ranking By Impact

1. Embedding memory/cache path optimization: 2m23s -> 1m8s
2. Validation scheduling optimization: 1m4s -> 51s
3. Preprocessing optimization (Triton + predict batch size tuning): 1m7s -> 1m4s

## Repro Context

- Task: offline BACH predict_fit
- Typical command: eva predict_fit --config configs/vision/pathology/offline/classification/bach.yaml
- Baseline config reference: configs/vision/pathology/offline/classification/bach0.yaml
- Profiling utility: profiling.sh (PyTorch Profiler + Nsight Systems, optional Nsight Compute)
