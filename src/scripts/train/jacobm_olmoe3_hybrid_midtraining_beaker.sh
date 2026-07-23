#!/usr/bin/env bash

set -euo pipefail

: "${OLMOE3_HYBRID_RUN_NAME:?OLMOE3_HYBRID_RUN_NAME must be set}"
: "${OLMOE3_MT_LOAD_PATH:?OLMOE3_MT_LOAD_PATH must be set}"
: "${BEAKER_TOKEN:?BEAKER_TOKEN must be set for the Beaker progress callback}"

SOURCE_REPO=${SOURCE_REPO:-/weka/oe-adapt-default/jacobm/olmoe3/olmo-ddp-migration/OLMo-core-moe-v2-core}
REPO=/tmp/OLMo-core
CREDENTIALS=/tmp/google/credentials.json

mkdir -p "$(dirname "${CREDENTIALS}")" /results
if [[ -n ${GANTRY_GOOGLE_CREDENTIALS:-} ]]; then
  printenv GANTRY_GOOGLE_CREDENTIALS > "${CREDENTIALS}"
  chmod 600 "${CREDENTIALS}"
  export GOOGLE_APPLICATION_CREDENTIALS="${CREDENTIALS}"
  unset GANTRY_GOOGLE_CREDENTIALS
fi

rm -rf "${REPO}"
mkdir -p "${REPO}/src/scripts/train/jacobm_olmoe_ladder/v2/models"
cp -a "${SOURCE_REPO}/src/olmo_core" "${REPO}/src/"
cp "${SOURCE_REPO}/src/scripts/train/jacobm_olmoe3_hybrid_scale.py" \
  "${REPO}/src/scripts/train/"
cp "${SOURCE_REPO}/src/scripts/train/jacobm_olmoe3_hybrid_midtraining.py" \
  "${REPO}/src/scripts/train/"
cp "${SOURCE_REPO}/src/scripts/train/jacobm_olmoe_ladder/v2/models/hybrid_wide.py" \
  "${REPO}/src/scripts/train/jacobm_olmoe_ladder/v2/models/"
cp "${SOURCE_REPO}/src/scripts/train/jacobm_olmoe_ladder/v2/models/geometry_matched_275m.py" \
  "${REPO}/src/scripts/train/jacobm_olmoe_ladder/v2/models/"
cp "${SOURCE_REPO}/src/scripts/train/jacobm_olmoe_ladder/v2/moe_v2_core_adapter.py" \
  "${REPO}/src/scripts/train/jacobm_olmoe_ladder/v2/"
cp -a "${SOURCE_REPO}/JACOBM_DDP_CONFIGS" "${REPO}/"

cd "${REPO}"
export PYTHONPATH="${REPO}/src"
export PYTHONUNBUFFERED=1
export CUDA_SCALE_LAUNCH_QUEUES=4x
export OLMO_SHARED_FS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
unset S3_PROFILE

torchrun --standalone --nproc-per-node="${OLMOE3_HYBRID_WORLD_SIZE:?}" \
  src/scripts/train/jacobm_olmoe3_hybrid_midtraining.py \
  train "${OLMOE3_HYBRID_RUN_NAME}" local \
  2>&1 | tee /results/train.log
