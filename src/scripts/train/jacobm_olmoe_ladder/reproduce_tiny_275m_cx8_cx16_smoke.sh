#!/usr/bin/env bash
set -euo pipefail

# Reproducibility record for Cx8/Cx16 high-microbatch smoke tests.
# By default this prints the commands without launching jobs.
# Set DRY_RUN=0 to submit them again.

DRY_RUN="${DRY_RUN:-1}"
SCRIPT="src/scripts/train/jacobm_olmoe_ladder/tiny_275m.py"
RUN_PREFIX="olmoe3-tiny-275m"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3}"
NUM_NODES=1
CHINCHILLA_MULTIPLE="${CHINCHILLA_MULTIPLE:-0.1}"

run_cmd() {
  printf 'Command:'
  printf ' %q' "$@"
  printf '\n'

  if [[ "${DRY_RUN}" == "0" ]]; then
    "$@"
  fi
}

launch_one() {
  local chinchilla="$1"
  local batch_tag="$2"
  local batch_seq="$3"
  local gpus="$4"
  local micro_bsz="$5"
  local lr="$6"
  local lr_tag="$7"
  local perf_tag="gpu${gpus}-ep1mb${micro_bsz}"
  local name="${RUN_PREFIX}-cx${chinchilla}-smoke-${batch_tag}-${perf_tag}-${lr_tag}"
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
        --save-folder="${CHECKPOINT_ROOT}/${name}" \
        --name="${name}" \
        --data-root=s3://ai2-llm \
        --lr="${lr}" \
        --chinchilla-multiple="${CHINCHILLA_MULTIPLE}" \
        --global-batch-size-seq="${batch_seq}" \
        --num-nodes="${NUM_NODES}" \
        --gpus-per-node="${gpus}" \
        --micro-batch-size="${micro_bsz}" \
        --ep-dim=1 \
        --tag="${lr_tag}-cx${chinchilla}-smoke-${batch_tag}-${perf_tag}"
}

launch_one 8 b768k 96 2 24 5e-4 lr5e-4
launch_one 16 b1m 128 2 32 5e-4 lr5e-4
