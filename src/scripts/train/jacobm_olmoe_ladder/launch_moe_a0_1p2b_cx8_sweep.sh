#!/usr/bin/env bash
set -euo pipefail

SCRIPT="src/scripts/train/jacobm_olmoe_ladder/tiny_275m.py"
RUN_PREFIX="olmoe3-moe-a0-1p2b-cx8"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3}"
LOG_DIR="${LOG_DIR:-/tmp/olmoe3-moe-a0-1p2b-cx8-launch-logs}"
JOB_CREATED_TIMEOUT_SECONDS="${JOB_CREATED_TIMEOUT_SECONDS:-240}"
NUM_NODES="${NUM_NODES:-4}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))
EP_DIM="${EP_DIM:-1}"
MICRO_BSZ="${MICRO_BSZ:-1}"
GLOBAL_BATCH_SIZE_SEQ="${GLOBAL_BATCH_SIZE_SEQ:-96}"
CHINCHILLA_MULTIPLE="${CHINCHILLA_MULTIPLE:-8}"
SWEEP_SUFFIX="${SWEEP_SUFFIX:-r1}"
EPHEMERAL_SAVE_INTERVAL="${EPHEMERAL_SAVE_INTERVAL:-500}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2000}"
LR_SPECS="${LR_SPECS:-2e-4:lr2e-4 4e-4:lr4e-4 8e-4:lr8e-4}"

mkdir -p "${LOG_DIR}"

launch_one() {
  local lr="$1"
  local lr_tag="$2"
  local perf_tag="gpu${TOTAL_GPUS}-ep${EP_DIM}mb${MICRO_BSZ}"
  local name="${RUN_PREFIX}-b768k-${perf_tag}-${lr_tag}-${SWEEP_SUFFIX}"
  local log_path="${LOG_DIR}/${name}.log"
  local common_beaker_args=(
    --cluster ai2/titan
    --nodes "${NUM_NODES}"
    --gpus "${GPUS_PER_NODE}"
    --weka oe-training-default
    --beaker-image tianhuat/olmo-core-torch211-2404-cu128
    --workspace ai2/OLMo-3-moe-experiments
    --budget ai2/oe-other
    --priority urgent
    --env OLMO_SYMM_VDEV2D_AUTO_BUILD=1
    --env-secret AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY WANDB_API_KEY=jacobm_WANDB_API_KEY
  )

  local cmd=(
    uv run --extra dev --extra beaker python -m olmo_core.launch.beaker
    --allow-dirty
    --name="${name}"
    "${common_beaker_args[@]}"
    --
    python "${SCRIPT}"
    --model-size=1p2b
    --save-folder="${CHECKPOINT_ROOT}/${name}"
    --name="${name}"
    --data-root=s3://ai2-llm
    --lr="${lr}"
    --chinchilla-multiple="${CHINCHILLA_MULTIPLE}"
    --global-batch-size-seq="${GLOBAL_BATCH_SIZE_SEQ}"
    --num-nodes="${NUM_NODES}"
    --gpus-per-node="${GPUS_PER_NODE}"
    --micro-batch-size="${MICRO_BSZ}"
    --ep-dim="${EP_DIM}"
    --ladder-evals
    --eval-task-set=fast
    --eval-interval="${EVAL_INTERVAL}"
    --save-interval=999999999
    --ephemeral-save-interval="${EPHEMERAL_SAVE_INTERVAL}"
    --no-pre-train-checkpoint
    --tag="${lr_tag}-1p2b-cx8-b768k-${perf_tag}-${SWEEP_SUFFIX}"
  )

  echo "Launching ${name}..."
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'

  "${cmd[@]}" >"${log_path}" 2>&1 &
  local pid=$!
  local deadline=$((SECONDS + JOB_CREATED_TIMEOUT_SECONDS))

  while (( SECONDS < deadline )); do
    if [[ -f "${log_path}" ]] && grep -q "job created" "${log_path}"; then
      sed -n '1,/job created/p' "${log_path}"
      kill "${pid}" 2>/dev/null || true
      wait "${pid}" 2>/dev/null || true
      echo "Detached local launcher for ${name}; Beaker job continues."
      return 0
    fi

    if ! kill -0 "${pid}" 2>/dev/null; then
      cat "${log_path}"
      wait "${pid}"
      return $?
    fi

    sleep 2
  done

  echo "Timed out waiting for Beaker job creation for ${name}; log follows:"
  cat "${log_path}"
  kill "${pid}" 2>/dev/null || true
  wait "${pid}" 2>/dev/null || true
  return 1
}

for spec in ${LR_SPECS}; do
  IFS=: read -r lr lr_tag <<<"${spec}"
  launch_one "${lr}" "${lr_tag}"
done
