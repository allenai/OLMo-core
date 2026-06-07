#!/usr/bin/env bash
set -euo pipefail

# Reproducibility record for 810M/1.2B MoE A0 smoke tests.
# By default this prints the commands without launching jobs.
# Set DRY_RUN=0 to submit them again.

DRY_RUN="${DRY_RUN:-1}"
SCRIPT="src/scripts/train/jacobm_olmoe_ladder/tiny_275m.py"
RUN_PREFIX="olmoe3-moe-a0"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3}"
NUM_NODES=1
GLOBAL_BATCH_SIZE_SEQ=32
CHINCHILLA_MULTIPLE="${CHINCHILLA_MULTIPLE:-0.02}"
SMOKE_SUFFIX="${SMOKE_SUFFIX:-r3}"

run_cmd() {
  printf 'Command:'
  printf ' %q' "$@"
  printf '\n'

  if [[ "${DRY_RUN}" == "0" ]]; then
    "$@"
  fi
}

launch_one() {
  local model_size="$1"
  local gpus="$2"
  local micro_bsz="$3"
  local ep_dim="$4"
  local lr="$5"
  local lr_tag="$6"
  local perf_tag="gpu${gpus}-ep${ep_dim}mb${micro_bsz}"
  local name="${RUN_PREFIX}-${model_size}-smoke-b256k-${perf_tag}-${lr_tag}-${SMOKE_SUFFIX}"
  local common_beaker_args=(
    --cluster ai2/titan
    --nodes "${NUM_NODES}"
    --gpus "${gpus}"
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
      --allow-dirty \
      --name="${name}" \
      "${common_beaker_args[@]}" \
      -- \
      python "${SCRIPT}" \
        --model-size="${model_size}" \
        --save-folder="${CHECKPOINT_ROOT}/${name}" \
        --name="${name}" \
        --data-root=s3://ai2-llm \
        --lr="${lr}" \
        --chinchilla-multiple="${CHINCHILLA_MULTIPLE}" \
        --global-batch-size-seq="${GLOBAL_BATCH_SIZE_SEQ}" \
        --num-nodes="${NUM_NODES}" \
        --gpus-per-node="${gpus}" \
        --micro-batch-size="${micro_bsz}" \
        --ep-dim="${ep_dim}" \
        --tag="${lr_tag}-${model_size}-smoke-b256k-${perf_tag}-${SMOKE_SUFFIX}"
}

launch_one 810m 4 8 1 5e-4 lr5e-4
launch_one 810m 4 4 1 5e-4 lr5e-4

launch_one 1p2b 4 4 1 3e-4 lr3e-4
launch_one 1p2b 4 4 2 3e-4 lr3e-4
