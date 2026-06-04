#!/usr/bin/env bash
set -euo pipefail

############################################
# User settings (override via env if needed)
############################################
: "${CONFIG:=configs/vision/pathology/offline/classification/bach.yaml}"
: "${GPU:=2,3,4,5}"
: "${MODEL_NAME:=universal/vit_small_patch16_224_dino}"

# Keep short for profiling baseline
: "${N_RUNS:=1}" # for profiling
: "${MAX_STEPS:=12500}"

# For per-slide wall-time in predict: use batch size 1
: "${PREDICT_BATCH_SIZE:=64}"

# Fit stage batch size (embeddings-based training)
: "${BATCH_SIZE:=256}"

# Data loader workers
: "${N_DATA_WORKERS:=4}"

# Output roots
# : "${OUTPUT_ROOT:=./logs/dino_vits16/offline/bach}"
: "${EMBEDDINGS_ROOT:=./data/embeddings/universal}"
: "${PROFILE_ROOT:=./profiles/bach}"

# Toggle heavy passes
: "${RUN_NSYS:=1}"
: "${RUN_NCU:=1}"

############################################
# Paths
############################################
PT_DIR="${PROFILE_ROOT}/pytorch"
NSYS_DIR="${PROFILE_ROOT}/nsys"
NCU_DIR="${PROFILE_ROOT}/ncu"

mkdir -p "${PT_DIR}" "${NSYS_DIR}" "${NCU_DIR}"

############################################
# Common env for all runs
############################################
export CUDA_VISIBLE_DEVICES="${GPU}"
export MODEL_NAME
export N_RUNS
export MAX_STEPS
export PREDICT_BATCH_SIZE
export BATCH_SIZE
export N_DATA_WORKERS
export OUTPUT_ROOT
export EMBEDDINGS_ROOT

echo "=== Profiling configuration ==="
echo "CONFIG=${CONFIG}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "MODEL_NAME=${MODEL_NAME}"
echo "N_RUNS=${N_RUNS}, MAX_STEPS=${MAX_STEPS}"
echo "PREDICT_BATCH_SIZE=${PREDICT_BATCH_SIZE}, BATCH_SIZE=${BATCH_SIZE}, N_DATA_WORKERS=${N_DATA_WORKERS}"
# echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "EMBEDDINGS_ROOT=${EMBEDDINGS_ROOT}"
echo "PROFILE_ROOT=${PROFILE_ROOT}"
echo

############################################
# 1) PyTorch Profiler pass
############################################
echo "=== [1/3] PyTorch Profiler: predict_fit ==="
python -m eva predict_fit \
  --config "${CONFIG}" \
  --trainer.init_args.profiler.class_path lightning.pytorch.profilers.PyTorchProfiler \
  --trainer.init_args.profiler.init_args.dirpath "${PT_DIR}" \
  --trainer.init_args.profiler.init_args.filename bach_predict_fit \
  --trainer.init_args.profiler.init_args.export_to_chrome true \
  --trainer.init_args.profiler.init_args.record_shapes true \
  --trainer.init_args.profiler.init_args.emit_nvtx true

############################################
# 2) Nsight Systems pass (end-to-end)
############################################
if [[ "${RUN_NSYS}" == "1" ]]; then
  command -v nsys >/dev/null 2>&1 || { echo "nsys not found; skip Nsight Systems"; RUN_NSYS=0; }
fi

if [[ "${RUN_NSYS}" == "1" ]]; then
  echo "=== [2/3] Nsight Systems: predict_fit end-to-end ==="
  rm -rf "${EMBEDDINGS_ROOT}/${MODEL_NAME}/bach"
  nsys profile \
    --trace=cuda,nvtx,osrt \
    --sample=cpu \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    -o "${NSYS_DIR}/bach_predict_fit" \
    python -m eva predict_fit --config "${CONFIG}"
fi

############################################
# 3) Nsight Compute pass (tensor-core metrics, optional)
#    Use predict only to keep runtime manageable.
############################################
if [[ "${RUN_NCU}" == "1" ]]; then
  command -v ncu >/dev/null 2>&1 || { echo "ncu not found; skip Nsight Compute"; RUN_NCU=0; }
fi

if [[ "${RUN_NCU}" == "1" ]]; then
  echo "=== [3/3] Nsight Compute: predict kernel metrics ==="
  rm -rf "${EMBEDDINGS_ROOT}/${MODEL_NAME}/bach"
  ncu \
    --target-processes all \
    --set full \
    --metrics \
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,smsp__inst_executed_pipe_tensor.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed \
    -o "${NCU_DIR}/bach_predict" \
    python -m eva predict_fit --config "${CONFIG}"
fi

############################################
# Output summary
############################################
cat <<EOF

Done.

Artifacts:
- PyTorch Profiler: ${PT_DIR}
- Nsight Systems:   ${NSYS_DIR}
- Nsight Compute:   ${NCU_DIR}

Quick readout:
1) PyTorch Profiler
   - Open TensorBoard:
     tensorboard --logdir "${PT_DIR}"
     - Step time with PREDICT_BATCH_SIZE=1 as approximate per-slide wall-time
     - Top ops by CUDA time
     - CPU gaps between steps (dataloader / I/O stalls)
     - Memory timeline

2) Nsight Systems (.nsys-rep)
   - Open in Nsight Systems UI
     - GPU busy vs idle intervals
     - CPU thread stalls
     - overlap of dataloading, embedding writing, and kernels
     - end-to-end predict->fit phase boundaries

3) Nsight Compute (.ncu-rep, if enabled)
   - Open in Nsight Compute UI
     - sm__pipe_tensor_cycles_active... (tensor-core activity proxy)
     - kernel-level throughput and hotspot kernels

EOF