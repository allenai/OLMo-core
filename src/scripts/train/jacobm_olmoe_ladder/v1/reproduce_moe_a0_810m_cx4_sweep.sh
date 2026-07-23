#!/usr/bin/env bash
set -euo pipefail

# Reproducibility record for the completed 810M MoE A0 Cx4 LR sweep.
# These jobs did not run in-loop ladder evals; final checkpoint evals were
# backfilled separately. By default this prints commands without launching jobs.
# Set DRY_RUN=0 to submit them again.

DRY_RUN="${DRY_RUN:-1}"
SCRIPT="src/scripts/train/jacobm_olmoe_ladder/tiny_275m.py"
RUN_PREFIX="olmoe3-moe-a0-810m-cx4"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3}"
NUM_NODES=1
GPUS="${GPUS:-8}"
EP_DIM=1
MICRO_BSZ=4
GLOBAL_BATCH_SIZE_SEQ=64
CHINCHILLA_MULTIPLE="${CHINCHILLA_MULTIPLE:-4}"
SWEEP_SUFFIX="${SWEEP_SUFFIX:-r1}"
EPHEMERAL_SAVE_INTERVAL="${EPHEMERAL_SAVE_INTERVAL:-500}"

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
  local perf_tag="gpu${GPUS}-ep${EP_DIM}mb${MICRO_BSZ}"
  local name="${RUN_PREFIX}-b512k-${perf_tag}-${lr_tag}-${SWEEP_SUFFIX}"
  local common_beaker_args=(
    --cluster ai2/titan
    --nodes "${NUM_NODES}"
    --gpus "${GPUS}"
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
        --model-size=810m \
        --save-folder="${CHECKPOINT_ROOT}/${name}" \
        --name="${name}" \
        --data-root=s3://ai2-llm \
        --lr="${lr}" \
        --chinchilla-multiple="${CHINCHILLA_MULTIPLE}" \
        --global-batch-size-seq="${GLOBAL_BATCH_SIZE_SEQ}" \
        --num-nodes="${NUM_NODES}" \
        --gpus-per-node="${GPUS}" \
        --micro-batch-size="${MICRO_BSZ}" \
        --ep-dim="${EP_DIM}" \
        --save-interval=999999999 \
        --ephemeral-save-interval="${EPHEMERAL_SAVE_INTERVAL}" \
        --no-pre-train-checkpoint \
        --tag="${lr_tag}-810m-cx4-b512k-${perf_tag}-${SWEEP_SUFFIX}"
}

launch_one 2e-4 lr2e-4
launch_one 4e-4 lr4e-4
launch_one 8e-4 lr8e-4
launch_one 1.6e-3 lr1.6e-3
