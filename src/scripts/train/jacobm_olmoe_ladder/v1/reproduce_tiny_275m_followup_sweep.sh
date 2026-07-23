#!/usr/bin/env bash
set -euo pipefail

# Reproducibility record for the follow-up tiny 275M-active MoE sweep.
# By default this prints the commands without launching jobs.
# Set DRY_RUN=0 to submit them again.

DRY_RUN="${DRY_RUN:-1}"
SCRIPT="src/scripts/train/jacobm_olmoe_ladder/tiny_275m.py"
RUN_PREFIX="olmoe3-tiny-275m"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3}"

COMMON_BEAKER_ARGS=(
  --cluster ai2/titan
  --nodes 1
  --gpus 8
  --weka oe-training-default
  --beaker-image tianhuat/olmo-core-torch211-2404-cu128
  --workspace ai2/OLMo-3-moe-experiments
  --budget ai2/oe-other
  --priority urgent
  --env OLMO_SYMM_VDEV2D_AUTO_BUILD=1
  --env-secret AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY WANDB_API_KEY=jacobm_WANDB_API_KEY
)

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
  local lr="$4"
  local lr_tag="$5"
  local name="${RUN_PREFIX}-cx${chinchilla}-${batch_tag}-${lr_tag}"

  run_cmd \
    uv run --extra dev --extra beaker python -m olmo_core.launch.beaker \
      --name="${name}" \
      "${COMMON_BEAKER_ARGS[@]}" \
      -- \
      python "${SCRIPT}" \
        --save-folder="${CHECKPOINT_ROOT}/${name}" \
        --name="${name}" \
        --data-root=s3://ai2-llm \
        --lr="${lr}" \
        --chinchilla-multiple="${chinchilla}" \
        --global-batch-size-seq="${batch_seq}" \
        --tag="${lr_tag}-cx${chinchilla}-${batch_tag}"
}

launch_one 2 b256k 32 5e-4 lr5e-4
launch_one 2 b256k 32 7e-4 lr7e-4
launch_one 1 b256k 32 4e-4 lr4e-4
launch_one 1 b256k 32 6e-4 lr6e-4
launch_one 1 b256k 32 7e-4 lr7e-4
launch_one 1 b256k 32 1e-3 lr1e-3
launch_one 1 b128k 16 5e-4 lr5e-4
launch_one 1 b128k 16 8e-4 lr8e-4
launch_one 1 b512k 64 5e-4 lr5e-4
launch_one 1 b512k 64 8e-4 lr8e-4
