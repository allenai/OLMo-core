#!/usr/bin/env bash
set -exuo pipefail

COMMAND=$1
shift

BASE_RUN_NAME=$1
shift

# Tell OLMo all ranks DO share the same filesystem for checkpoints.
export OLMO_SHARED_FS=1
# # Tell OLMo all ranks do NOT share the same filesystem for checkpoints.
# unset OLMO_SHARED_FS

# Job details
RUN_NAME=${BASE_RUN_NAME}-$(date -u +"%Y%m%d_%H%M%S")
SAVE_FOLDER=/mnt/checkpoints/shanea/checkpoints/OLMo-large/$RUN_NAME
mkdir -p $SAVE_FOLDER

torchrun \
  --nnodes $SLURM_NNODES \
  --nproc-per-node $SLURM_GPUS_PER_NODE \
  --rdzv_id $SLURM_JOB_ID \
  --node_rank $SLURM_PROCID \
  --rdzv_backend c10d \
  --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
  src/scripts/train/OLMo2-32B.py $COMMAND $RUN_NAME --trainer.save_folder=$SAVE_FOLDER ${@}
