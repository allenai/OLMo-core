#!/usr/bin/env bash

set -euo pipefail

: "${OLMOE3_HYBRID_RUN_NAME:?OLMOE3_HYBRID_RUN_NAME must be set}"
: "${OLMOE3_HYBRID_MODEL_SIZE:?OLMOE3_HYBRID_MODEL_SIZE must be set}"

SOURCE_REPO=${SOURCE_REPO:-/weka/oe-adapt-default/jacobm/olmoe3/olmo-ddp-migration/OLMo-core-moe-v2-core}
REPO=/tmp/OLMo-core

rm -rf "${REPO}"
mkdir -p "${REPO}/src/scripts/train/jacobm_olmoe_ladder/v2/models" /results
cp -a "${SOURCE_REPO}/src/olmo_core" "${REPO}/src/"
cp "${SOURCE_REPO}/src/scripts/train/jacobm_olmoe3_hybrid_scale.py" \
  "${REPO}/src/scripts/train/"
cp "${SOURCE_REPO}/src/scripts/train/jacobm_olmoe_ladder/v2/models/hybrid_wide.py" \
  "${REPO}/src/scripts/train/jacobm_olmoe_ladder/v2/models/"
cp "${SOURCE_REPO}/src/scripts/train/jacobm_olmoe_ladder/v2/models/geometry_matched_275m.py" \
  "${REPO}/src/scripts/train/jacobm_olmoe_ladder/v2/models/"
cp "${SOURCE_REPO}/src/scripts/train/jacobm_olmoe_ladder/v2/models/geometry_matched_scale.py" \
  "${REPO}/src/scripts/train/jacobm_olmoe_ladder/v2/models/"
cp "${SOURCE_REPO}/src/scripts/train/jacobm_olmoe_ladder/v2/moe_v2_core_adapter.py" \
  "${REPO}/src/scripts/train/jacobm_olmoe_ladder/v2/"
cp -a "${SOURCE_REPO}/JACOBM_DDP_CONFIGS" "${REPO}/"

cd "${REPO}"
export PYTHONPATH="${REPO}/src"
if [[ -n ${OLMOE3_HYBRID_FLA_OVERLAY:-} ]]; then
  export PYTHONPATH="${OLMOE3_HYBRID_FLA_OVERLAY}:${PYTHONPATH}"
fi
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

SUBCOMMAND=${OLMOE3_HYBRID_SUBCOMMAND:-train}
LOG_NAME=train.log
if [[ "${SUBCOMMAND}" == "eval_checkpoints" ]]; then
  LOG_NAME=eval.log
fi

torchrun --standalone --nproc-per-node="${OLMOE3_HYBRID_WORLD_SIZE:?}" \
  src/scripts/train/jacobm_olmoe3_hybrid_scale.py \
  "${SUBCOMMAND}" "${OLMOE3_HYBRID_RUN_NAME}" local \
  2>&1 | tee "/results/${LOG_NAME}"
