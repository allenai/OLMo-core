#!/usr/bin/env bash
set -euo pipefail

SCRIPT="src/scripts/train/jacobm_olmoe_ladder/experiments/integration/integration_ladder.py"
RUN_PREFIX="${RUN_PREFIX:-int-275m}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/integration}"
LOG_DIR="${LOG_DIR:-/tmp/olmoe3-integration-275m-launch-logs}"
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
INTEGRATION_VARIANTS="${INTEGRATION_VARIANTS:-wide_256e8k deep_256e8k}"
DATA_MULTIPLES="${DATA_MULTIPLES:-1 2 4 8}"
LRS="${LRS:-8e-4 1.6e-3 3.2e-3}"

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

integration_tag_for() {
  case "$1" in
    wide_256e8k) echo intw256e8k ;;
    deep_256e8k) echo intd256e8k ;;
    *) echo "Unknown integration variant: $1" >&2; return 1 ;;
  esac
}

lr_tag_for() {
  case "$1" in
    8e-4) echo lr8e-4 ;;
    1.6e-3) echo lr1.6e-3 ;;
    3.2e-3) echo lr3.2e-3 ;;
    *) echo "Unknown LR: $1" >&2; return 1 ;;
  esac
}

batch_for_cx() {
  case "$1" in
    1) echo "32 2 8 b256k" ;;
    2) echo "48 2 8 b384k" ;;
    4) echo "64 4 8 b512k" ;;
    8) echo "96 8 4 b768k" ;;
    *) echo "Unknown Chinchilla multiple: $1" >&2; return 1 ;;
  esac
}

launch_one() {
  local integration_variant="$1"
  local integration_tag="$2"
  local cx="$3"
  local lr="$4"
  local lr_tag="$5"

  read -r global_batch_size_seq gpus micro_bsz batch_tag < <(batch_for_cx "${cx}")
  local denom=$((NUM_NODES * gpus * micro_bsz))
  if (( global_batch_size_seq % denom != 0 )); then
    echo "Invalid batch settings for ${integration_variant} Cx${cx}: global_batch_size_seq=${global_batch_size_seq} is not divisible by nodes*gpus*micro_bsz=${denom}" >&2
    exit 1
  fi

  local name="${RUN_PREFIX}-cx${cx}-${integration_tag}-${lr_tag}-${SWEEP_SUFFIX}"
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
    --model-size=275m
    --integration-config="${integration_variant}"
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
    --tag="${integration_tag}-cx${cx}-${lr_tag}-${SWEEP_SUFFIX}"
    --wandb-tag=exp_integration
    --wandb-tag="${integration_tag}"
    --wandb-tag=275m
    --wandb-tag="cx${cx}"
    --wandb-tag="${lr_tag}"
    --wandb-tag="${systems_tag}"
    --wandb-tag=compile-on
    --wandb-tag=baseline-centered
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

for integration_variant in ${INTEGRATION_VARIANTS}; do
  integration_tag="$(integration_tag_for "${integration_variant}")"
  for cx in ${DATA_MULTIPLES}; do
    for lr in ${LRS}; do
      lr_tag="$(lr_tag_for "${lr}")"
      launch_one "${integration_variant}" "${integration_tag}" "${cx}" "${lr}" "${lr_tag}"
    done
  done
done
