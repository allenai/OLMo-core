#!/usr/bin/env bash
set -euo pipefail

# Reproducibility record for the first full 1.2B MoE A0 Cx1 LR sweep.
# By default this prints commands without launching jobs.
# Set DRY_RUN=0 to submit them after LR_SPECS is finalized.

DRY_RUN="${DRY_RUN:-1}"
SCRIPT="src/scripts/train/jacobm_olmoe_ladder/tiny_275m.py"
RUN_PREFIX="olmoe3-moe-a0-1p2b-cx1"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3}"
NUM_NODES=1
GPUS="${GPUS:-8}"
EP_DIM=1
MICRO_BSZ=2
GLOBAL_BATCH_SIZE_SEQ=32
CHINCHILLA_MULTIPLE="${CHINCHILLA_MULTIPLE:-1}"
SWEEP_SUFFIX="${SWEEP_SUFFIX:-r1}"
EPHEMERAL_SAVE_INTERVAL="${EPHEMERAL_SAVE_INTERVAL:-500}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2000}"
LR_SPECS="${LR_SPECS:-}"

if [[ -z "${LR_SPECS}" ]]; then
  echo "LR_SPECS is required, e.g. LR_SPECS='2e-4:lr2e-4 4e-4:lr4e-4' $0" >&2
  exit 2
fi

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
  local name="${RUN_PREFIX}-b256k-${perf_tag}-${lr_tag}-${SWEEP_SUFFIX}"
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
        --model-size=1p2b \
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
        --ladder-evals \
        --eval-task-set=fast \
        --eval-interval="${EVAL_INTERVAL}" \
        --save-interval=999999999 \
        --ephemeral-save-interval="${EPHEMERAL_SAVE_INTERVAL}" \
        --no-pre-train-checkpoint \
        --tag="${lr_tag}-1p2b-cx1-b256k-${perf_tag}-${SWEEP_SUFFIX}"
}

for spec in ${LR_SPECS}; do
  lr="${spec%%:*}"
  lr_tag="${spec#*:}"
  if [[ -z "${lr}" || -z "${lr_tag}" || "${lr}" == "${lr_tag}" ]]; then
    echo "Invalid LR spec '${spec}', expected lr:tag" >&2
    exit 2
  fi
  launch_one "${lr}" "${lr_tag}"
done
