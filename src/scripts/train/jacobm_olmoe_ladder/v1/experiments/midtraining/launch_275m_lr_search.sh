#!/usr/bin/env bash
set -euo pipefail

SCRIPT="src/scripts/train/jacobm_olmoe_ladder/experiments/midtraining/midtraining_ladder.py"
RUN_PREFIX="${RUN_PREFIX:-mt-275m}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/midtraining}"
LOG_DIR="${LOG_DIR:-/tmp/olmoe3-midtraining-275m-launch-logs}"
JOB_CREATED_TIMEOUT_SECONDS="${JOB_CREATED_TIMEOUT_SECONDS:-240}"
CLUSTER="${CLUSTER:-ai2/titan}"
BEAKER_IMAGE="${BEAKER_IMAGE:-tianhuat/olmo-core-torch211-2404-cu128}"
WORKSPACE="${WORKSPACE:-ai2/OLMo-3-moe-experiments}"
BUDGET="${BUDGET:-ai2/oe-other}"
PRIORITY="${PRIORITY:-urgent}"
PREEMPTIBLE="${PREEMPTIBLE:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SWEEP_SUFFIX="${SWEEP_SUFFIX:-r1}"
MIDTRAIN_MAX_TOKENS="${MIDTRAIN_MAX_TOKENS:-100000000000}"
EPHEMERAL_SAVE_INTERVAL="${EPHEMERAL_SAVE_INTERVAL:-500}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2000}"
MODEL_SIZE="${MODEL_SIZE:-275m}"
GLOBAL_BATCH_SIZE_SEQ="${GLOBAL_BATCH_SIZE_SEQ:-128}"
GPUS="${GPUS:-4}"
NUM_NODES="${NUM_NODES:-1}"
MICRO_BSZ="${MICRO_BSZ:-8}"
EP_DIM="${EP_DIM:-1}"
LRS="${LRS:-2e-4 4e-4 8e-4 1.6e-3}"
SOURCE_TAG="${SOURCE_TAG:-}"
LOAD_PATH="${LOAD_PATH:-}"

if [[ -z "${LOAD_PATH}" ]]; then
  echo "LOAD_PATH must point at the pretrained OLMo-core checkpoint/folder to midtrain from" >&2
  exit 1
fi
if [[ -z "${SOURCE_TAG}" ]]; then
  echo "SOURCE_TAG must be a short checkpoint/source label for run names, e.g. baseline-cx8" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

denom=$((NUM_NODES * GPUS * MICRO_BSZ))
if (( GLOBAL_BATCH_SIZE_SEQ % denom != 0 )); then
  echo "Invalid batch settings: GLOBAL_BATCH_SIZE_SEQ=${GLOBAL_BATCH_SIZE_SEQ} is not divisible by nodes*gpus*MICRO_BSZ=${denom}" >&2
  exit 1
fi

common_beaker_args=(
  --cluster "${CLUSTER}"
  --nodes "${NUM_NODES}"
  --gpus "${GPUS}"
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

lr_tag_for() {
  case "$1" in
    2e-4) echo lr2e-4 ;;
    3e-4) echo lr3e-4 ;;
    4e-4) echo lr4e-4 ;;
    6e-4) echo lr6e-4 ;;
    8e-4) echo lr8e-4 ;;
    1.2e-3) echo lr1.2e-3 ;;
    1.6e-3) echo lr1.6e-3 ;;
    2e-3) echo lr2e-3 ;;
    *) echo "lr${1}" | tr . p ;;
  esac
}

launch_one() {
  local lr="$1"
  local lr_tag
  lr_tag="$(lr_tag_for "${lr}")"
  local batch_tag="b$((GLOBAL_BATCH_SIZE_SEQ * 8192 / 1024))k"
  local name="${RUN_PREFIX}-${SOURCE_TAG}-${lr_tag}-${SWEEP_SUFFIX}"
  local log_path="${LOG_DIR}/${name}.log"
  local systems_tag="${batch_tag}-gpu${GPUS}-ep${EP_DIM}mb${MICRO_BSZ}"

  local cmd=(
    uv run --extra dev --extra beaker python -m olmo_core.launch.beaker
    --allow-dirty
    --name="${name}"
    "${common_beaker_args[@]}"
    --
    "${PYTHON_BIN}" "${SCRIPT}"
    --model-size="${MODEL_SIZE}"
    --compile
    --save-folder="${CHECKPOINT_ROOT}/${name}"
    --name="${name}"
    --data-root=s3://ai2-llm
    --load-path="${LOAD_PATH}"
    --lr="${lr}"
    --midtrain-max-tokens="${MIDTRAIN_MAX_TOKENS}"
    --global-batch-size-seq="${GLOBAL_BATCH_SIZE_SEQ}"
    --num-nodes="${NUM_NODES}"
    --gpus-per-node="${GPUS}"
    --micro-batch-size="${MICRO_BSZ}"
    --ep-dim="${EP_DIM}"
    --eval-task-set=fast
    --eval-interval="${EVAL_INTERVAL}"
    --save-interval=999999999
    --ephemeral-save-interval="${EPHEMERAL_SAVE_INTERVAL}"
    --no-pre-train-checkpoint
    --tag="midtrain-${SOURCE_TAG}-${lr_tag}-${SWEEP_SUFFIX}"
    --wandb-tag=exp_midtraining
    --wandb-tag="${MODEL_SIZE}"
    --wandb-tag="${SOURCE_TAG}"
    --wandb-tag="${lr_tag}"
    --wandb-tag="${systems_tag}"
    --wandb-tag=compile-on
    --wandb-tag=weight-only-load
    --wandb-tag=titan
  )

  echo "Launching ${name}..."
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '
'

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

for lr in ${LRS}; do
  launch_one "${lr}"
done
