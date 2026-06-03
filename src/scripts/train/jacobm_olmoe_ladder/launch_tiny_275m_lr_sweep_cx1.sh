#!/usr/bin/env bash
set -euo pipefail

SCRIPT="src/scripts/train/jacobm_olmoe_ladder/tiny_275m.py"
RUN_PREFIX="olmoe3-tiny-275m-cx1"
SAVE_ROOT="/weka/oe-training-default/ai2-llm/checkpoints/${USER}"

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

launch_one() {
  local lr="$1"
  local lr_tag="$2"
  local name="${RUN_PREFIX}-${lr_tag}"

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
      --tag="${lr_tag}-cx1"
}

launch_one "1e-4" "lr1e-4"
launch_one "3e-4" "lr3e-4"
launch_one "8e-4" "lr8e-4"
launch_one "1.2e-3" "lr1.2e-3"
