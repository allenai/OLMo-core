#!/usr/bin/env bash
set -euo pipefail

SCRIPT="src/scripts/train/jacobm_olmoe_ladder/experiments/integration/integration_ladder.py"
RUN_PREFIX="${RUN_PREFIX:-int-480m}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/integration}"
LOG_DIR="${LOG_DIR:-/tmp/olmoe3-integration-480m-wide-launch-logs}"
JOB_CREATED_TIMEOUT_SECONDS="${JOB_CREATED_TIMEOUT_SECONDS:-240}"
NUM_NODES="${NUM_NODES:-1}"
EP_DIM=1
CLUSTER="${CLUSTER:-ai2/titan}"
BEAKER_IMAGE="${BEAKER_IMAGE:-tianhuat/olmo-core-torch211-2404-cu128}"
WORKSPACE="${WORKSPACE:-ai2/OLMo-3-moe-experiments}"
BUDGET="${BUDGET:-ai2/oe-other}"
PRIORITY="${PRIORITY:-urgent}"
PREEMPTIBLE="${PREEMPTIBLE:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SWEEP_SUFFIX="${SWEEP_SUFFIX:-r1}"
EPHEMERAL_SAVE_INTERVAL="${EPHEMERAL_SAVE_INTERVAL:-500}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2000}"
INTEGRATION_VARIANT="${INTEGRATION_VARIANT:-wide_256e8k}"
INTEGRATION_TAG="${INTEGRATION_TAG:-intw256e8k}"
CX_VALUES="${CX_VALUES:-1 2 4 8}"

mkdir -p "${LOG_DIR}"

common_beaker_args=(
  --cluster "${CLUSTER}"
  --nodes "${NUM_NODES}"
  --weka oe-training-default
  --beaker-image "${BEAKER_IMAGE}"
  --workspace "${WORKSPACE}"
  --budget "${BUDGET}"
  --priority "${PRIORITY}"
  --env OLMO_SYMM_VDEV2D_AUTO_BUILD=1
  --env-secret AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY WANDB_API_KEY=jacobm_WANDB_API_KEY
)
if [[ "${PREEMPTIBLE}" == "1" ]]; then
  common_beaker_args+=(--preemptible)
fi

should_launch_cx() {
  local want="$1"
  for cx_value in ${CX_VALUES}; do
    if [[ "${cx_value}" == "${want}" ]]; then
      return 0
    fi
  done
  return 1
}

launch_one() {
  local cx="$1"
  local batch_tag="$2"
  local global_batch_size_seq="$3"
  local gpus="$4"
  local micro_bsz="$5"
  local lr="$6"
  local lr_tag="$7"

  if ! should_launch_cx "${cx}"; then
    echo "Skipping 480m ${INTEGRATION_TAG} Cx${cx}; CX_VALUES=${CX_VALUES}"
    return 0
  fi

  local denom=$((NUM_NODES * gpus * micro_bsz))
  if (( global_batch_size_seq % denom != 0 )); then
    echo "Invalid batch settings for 480m ${INTEGRATION_TAG} Cx${cx}: global_batch_size_seq=${global_batch_size_seq} is not divisible by nodes*gpus*micro_bsz=${denom}" >&2
    exit 1
  fi

  local name="${RUN_PREFIX}-cx${cx}-${INTEGRATION_TAG}-${lr_tag}-${SWEEP_SUFFIX}"
  local log_path="${LOG_DIR}/${name}.log"
  local systems_tag="${batch_tag}-gpu${gpus}-ep${EP_DIM}mb${micro_bsz}"

  local cmd=(
    uv run --extra dev --extra beaker python -m olmo_core.launch.beaker
    --allow-dirty
    --name="${name}"
    --gpus "${gpus}"
    "${common_beaker_args[@]}"
    --
    "${PYTHON_BIN}" "${SCRIPT}"
    --model-size=480m
    --integration-config="${INTEGRATION_VARIANT}"
    --compile
    --save-folder="${CHECKPOINT_ROOT}/${name}"
    --name="${name}"
    --data-root=s3://ai2-llm
    --lr="${lr}"
    --chinchilla-multiple="${cx}"
    --global-batch-size-seq="${global_batch_size_seq}"
    --num-nodes="${NUM_NODES}"
    --gpus-per-node="${gpus}"
    --micro-batch-size="${micro_bsz}"
    --ep-dim="${EP_DIM}"
    --ladder-evals
    --eval-task-set=fast
    --eval-interval="${EVAL_INTERVAL}"
    --save-interval=999999999
    --ephemeral-save-interval="${EPHEMERAL_SAVE_INTERVAL}"
    --no-pre-train-checkpoint
    --tag="${INTEGRATION_TAG}-480m-cx${cx}-${lr_tag}-${SWEEP_SUFFIX}"
    --wandb-tag=exp_integration
    --wandb-tag="${INTEGRATION_TAG}"
    --wandb-tag=480m
    --wandb-tag="cx${cx}"
    --wandb-tag="${batch_tag}"
    --wandb-tag="${lr_tag}"
    --wandb-tag="${systems_tag}"
    --wandb-tag=compile-on
    --wandb-tag=baseline-best-observed
    --wandb-tag=promoted-single-point
    --wandb-tag=titan
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

launch_one 1 b256k 32 4 4 1.2e-3 lr1.2e-3
launch_one 2 b384k 48 4 4 9e-4 lr9e-4
launch_one 4 b512k 64 4 4 8e-4 lr8e-4
launch_one 8 b768k 96 8 4 8e-4 lr8e-4
