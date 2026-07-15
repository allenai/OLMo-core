#!/usr/bin/env bash

set -euo pipefail

: "${OLMOE3_HYBRID_RUN_NAME:?OLMOE3_HYBRID_RUN_NAME must be set}"
: "${OLMOE3_HYBRID_MODEL_SIZE:?OLMOE3_HYBRID_MODEL_SIZE must be set}"

SOURCE_REPO=${SOURCE_REPO:-/weka/oe-adapt-default/jacobm/olmoe3/olmo-ddp-migration/OLMo-core}
REPO=/tmp/OLMo-core

rm -rf "${REPO}"
mkdir -p "${REPO}/src/scripts/train/jacobm_olmoe_ladder/v2/models" /results
cp -a "${SOURCE_REPO}/src/olmo_core" "${REPO}/src/"
cp "${SOURCE_REPO}/src/scripts/train/jacobm_olmoe3_hybrid_scale.py" \
  "${REPO}/src/scripts/train/"
cp "${SOURCE_REPO}/src/scripts/train/jacobm_olmoe_ladder/v2/models/hybrid_wide.py" \
  "${REPO}/src/scripts/train/jacobm_olmoe_ladder/v2/models/"
cp -a "${SOURCE_REPO}/JACOBM_DDP_CONFIGS" "${REPO}/"

cd "${REPO}"
export PYTHONPATH="${REPO}/src"
export PYTHONUNBUFFERED=1
export CUDA_SCALE_LAUNCH_QUEUES=4x
export OLMO_SHARED_FS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
unset S3_PROFILE

if (( ${OLMOE3_HYBRID_EP_SIZE:-1} > 1 )) && \
  [[ ${OLMOE3_HYBRID_EP_PATH:-rowwise_nvshmem} == rowwise_nvshmem ]]; then
  python -m olmo_core.kernels.build_symm_mem_vdev2d_ext \
    --inplace --backend cmake
fi

torchrun --standalone --nproc-per-node="${OLMOE3_HYBRID_WORLD_SIZE:?}" \
  src/scripts/train/jacobm_olmoe3_hybrid_scale.py \
  train "${OLMOE3_HYBRID_RUN_NAME}" local \
  2>&1 | tee /results/train.log
