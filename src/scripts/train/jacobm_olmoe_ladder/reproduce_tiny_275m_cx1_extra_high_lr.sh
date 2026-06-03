#!/usr/bin/env bash
set -euo pipefail

# Reproducibility record for the Cx1 extra high-side LR probes.
# By default this prints the commands without launching jobs.
# Set DRY_RUN=0 to submit them again.

DRY_RUN="${DRY_RUN:-1}"
SCRIPT="src/scripts/train/jacobm_olmoe_ladder/tiny_275m.py"
RUN_PREFIX="olmoe3-tiny-275m"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3}"
NUM_NODES=1

COMMON_BEAKER_ARGS=(
  --cluster ai2/titan
  --nodes "${NUM_NODES}"
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
  local name="${RUN_PREFIX}-cx1-b256k-${lr_tag}"

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
        --chinchilla-multiple=1 \
        --global-batch-size-seq=32 \
        --num-nodes="${NUM_NODES}" \
        --tag="${lr_tag}-cx1-b256k"
}

launch_one 3e-3 lr3e-3
launch_one 5e-3 lr5e-3
