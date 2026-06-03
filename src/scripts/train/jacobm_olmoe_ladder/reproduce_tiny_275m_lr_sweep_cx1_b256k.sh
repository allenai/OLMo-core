#!/usr/bin/env bash
set -euo pipefail

# Reproducibility record for the tiny 275M-active MoE Cx1 LR sweep at 256k tokens/step.
#
# Run from the repository root.
# By default this prints the commands without launching jobs:
#
#   src/scripts/train/jacobm_olmoe_ladder/reproduce_tiny_275m_lr_sweep_cx1_b256k.sh
#
# To actually launch the sweep again:
#
#   DRY_RUN=0 src/scripts/train/jacobm_olmoe_ladder/reproduce_tiny_275m_lr_sweep_cx1_b256k.sh

DRY_RUN="${DRY_RUN:-1}"

SCRIPT="src/scripts/train/jacobm_olmoe_ladder/tiny_275m.py"
RUN_PREFIX="olmoe3-tiny-275m-cx1-b256k"
SAVE_ROOT="/weka/oe-training-default/ai2-llm/checkpoints/${USER}"
GLOBAL_BATCH_SIZE_SEQ=32

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
  local lr="$1"
  local lr_tag="$2"
  local name="${RUN_PREFIX}-${lr_tag}"

  run_cmd \
    uv run --extra dev --extra beaker python -m olmo_core.launch.beaker \
      --name="${name}" \
      "${COMMON_BEAKER_ARGS[@]}" \
      -- \
      python "${SCRIPT}" \
        --save-folder="${SAVE_ROOT}/${name}" \
        --name="${name}" \
        --data-root=s3://ai2-llm \
        --lr="${lr}" \
        --chinchilla-multiple=1 \
        --global-batch-size-seq="${GLOBAL_BATCH_SIZE_SEQ}" \
        --tag="${lr_tag}-cx1-b256k"
}

launch_one "3e-4" "lr3e-4"
launch_one "5e-4" "lr5e-4"
launch_one "8e-4" "lr8e-4"
launch_one "1.2e-3" "lr1.2e-3"
