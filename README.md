# EVA Fork: BACH Speedup Study

This fork is a small project focused on transformer fine-tuning speedup for offline BACH evaluation.
It keeps the EVA pipeline and adds practical runtime optimizations plus profiling workflows.

**Total wall-time speedup: 2m23s -> 51s (-64.3% time)**

## Installation

Install from PyPI:

```sh
# core
pip install kaiko-eva

# with vision support
pip install 'kaiko-eva[vision]'

# with language support
pip install 'kaiko-eva[language]'

# with multimodal support
pip install 'kaiko-eva[multimodal]'

# full install
pip install 'kaiko-eva[all]'
```

Or install the latest main branch:

```sh
pip install "kaiko-eva[all] @ git+https://github.com/kaiko-ai/eva.git"
```

Validate installation:

```sh
eva --version
```

## Quick Start (BACH)

Use the optimized config:

```sh
DOWNLOAD_DATA=true \
MODEL_NAME=universal/vit_small_patch16_224_dino \
eva predict_fit --config configs/vision/pathology/offline/classification/bach.yaml
```

Run the original baseline config:

```sh
DOWNLOAD_DATA=true \
MODEL_NAME=universal/vit_small_patch16_224_dino \
eva predict_fit --config configs/vision/pathology/offline/classification/bach0.yaml
```

Config notes:
- Optimized: configs/vision/pathology/offline/classification/bach.yaml
- Original: configs/vision/pathology/offline/classification/bach0.yaml

## Reproducible Weights & Biases Logging

Install and authenticate:

```sh
pip install wandb
wandb login
```

Then run with explicit WandbLogger injection:

```sh
eva predict_fit --config configs/vision/pathology/offline/classification/bach.yaml --trainer.init_args.logger='[{"class_path":"lightning.pytorch.loggers.WandbLogger","init_args":{"project":"'"eva"'","save_dir":"logs","log_model":false}}]'
```

You can replace the config path with bach0.yaml to compare optimized vs original behavior.

## Profiling Script

Use profiling.sh to run profiling for the BACH predict_fit flow.
It includes:
- PyTorch Profiler pass
- Nsight Systems pass
- optional Nsight Compute pass

Example:

```sh
bash profiling.sh
```

## Speedup Findings

A ranked bottleneck summary, fixes, and measured results are documented in [issue.md](issue.md).
