#!/usr/bin/env bash

set -euo pipefail

: "${OLMOE3_PORT_RUN_NAME:?OLMOE3_PORT_RUN_NAME must be set}"
: "${OLMOE3_PORT_WORLD_SIZE:?OLMOE3_PORT_WORLD_SIZE must be set}"

CANDIDATE_REPO=${CANDIDATE_REPO:-/weka/oe-adapt-default/jacobm/olmoe3/olmo-ddp-migration/OLMo-core-moe-v2-core-validation}
mkdir -p /results
export PYTHONPATH="${CANDIDATE_REPO}/src"
export PYTHONUNBUFFERED=1
export CUDA_SCALE_LAUNCH_QUEUES=4x
export OLMO_SHARED_FS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
unset S3_PROFILE

torchrun --standalone --nproc-per-node="${OLMOE3_PORT_WORLD_SIZE}" \
  "${CANDIDATE_REPO}/src/scripts/train/jacobm_moe_v2_port_validation/train.py" \
  train "${OLMOE3_PORT_RUN_NAME}" local \
  2>&1 | tee /results/train.log
