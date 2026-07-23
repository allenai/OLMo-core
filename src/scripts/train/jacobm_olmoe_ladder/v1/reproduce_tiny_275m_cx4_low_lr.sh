#!/usr/bin/env bash
set -euo pipefail

# Reproducibility record for the Cx4 low-side LR probes.
# By default this prints the commands without launching jobs.
# Set DRY_RUN=0 to submit them again.

DRY_RUN="${DRY_RUN:-1}"
SCRIPT="src/scripts/train/jacobm_olmoe_ladder/tiny_275m.py"
RUN_PREFIX="olmoe3-tiny-275m"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3}"
NUM_NODES=1
NUM_GPUS=4
MICRO_BSZ=16

run_cmd() {
  printf 'Command:'
  printf ' %q' "$@"
  printf '\n'

  if [[ "${DRY_RUN}" == "0" ]]; then
    "$@"
  fi
}

launch_one() {
  local lr="$1"
  local lr_tag="$2"
  local name="${RUN_PREFIX}-cx4-b512k-gpu${NUM_GPUS}-ep1mb${MICRO_BSZ}-${lr_tag}"
  local common_beaker_args=(
    --cluster ai2/titan
    --nodes "${NUM_NODES}"
    --gpus "${NUM_GPUS}"
    --weka oe-training-default
    --beaker-image tianhuat/olmo-core-torch211-2404-cu128
    --workspace ai2/OLMo-3-moe-experiments
    --budget ai2/oe-other
    --priority urgent
    --env OLMO_SYMM_VDEV2D_AUTO_BUILD=1
    --env-secret AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY WANDB_API_KEY=jacobm_WANDB_API_KEY
  )

  run_cmd \
    uv run --extra dev --extra beaker python -m olmo_core.launch.beaker \
      --name="${name}" \
      "${common_beaker_args[@]}" \
      -- \
      python "${SCRIPT}" \
        --save-folder="${CHECKPOINT_ROOT}/${name}" \
        --name="${name}" \
        --data-root=s3://ai2-llm \
        --lr="${lr}" \
        --chinchilla-multiple=4 \
        --global-batch-size-seq=64 \
        --num-nodes="${NUM_NODES}" \
        --gpus-per-node="${NUM_GPUS}" \
        --micro-batch-size="${MICRO_BSZ}" \
        --ep-dim=1 \
        --tag="${lr_tag}-cx4-b512k-gpu${NUM_GPUS}-ep1mb${MICRO_BSZ}"
}

launch_one 5e-4 lr5e-4
launch_one 7e-4 lr7e-4
