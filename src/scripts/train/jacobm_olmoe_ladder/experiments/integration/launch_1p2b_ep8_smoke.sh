#!/usr/bin/env bash
set -euo pipefail

SCRIPT="src/scripts/train/jacobm_olmoe_ladder/experiments/integration/integration_ladder.py"
RUN_PREFIX="${RUN_PREFIX:-int-smoke-1p2b-cx8}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/integration_smoke}"
LOG_DIR="${LOG_DIR:-/tmp/olmoe3-integration-1p2b-ep8-smoke-launch-logs}"
JOB_CREATED_TIMEOUT_SECONDS="${JOB_CREATED_TIMEOUT_SECONDS:-240}"
NUM_NODES="${NUM_NODES:-1}"
GPUS="${GPUS:-8}"
EP_DIM="${EP_DIM:-8}"
MICRO_BSZ="${MICRO_BSZ:-3}"
GLOBAL_BATCH_SIZE_SEQ="${GLOBAL_BATCH_SIZE_SEQ:-96}"
CLUSTER="${CLUSTER:-ai2/titan}"
BEAKER_IMAGE="${BEAKER_IMAGE:-tianhuat/olmo-core-torch211-2404-cu128}"
WORKSPACE="${WORKSPACE:-ai2/OLMo-3-moe-experiments}"
BUDGET="${BUDGET:-ai2/oe-other}"
PRIORITY="${PRIORITY:-urgent}"
PREEMPTIBLE="${PREEMPTIBLE:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SWEEP_SUFFIX="${SWEEP_SUFFIX:-ep8mb3-r1}"
EPHEMERAL_SAVE_INTERVAL="${EPHEMERAL_SAVE_INTERVAL:-500}"
EVAL_INTERVAL="${EVAL_INTERVAL:-999999}"
CHINCHILLA_MULTIPLE="${CHINCHILLA_MULTIPLE:-0.02}"
LR="${LR:-4e-4}"
LR_TAG="${LR_TAG:-lr4e-4}"
INTEGRATION_VARIANTS="${INTEGRATION_VARIANTS:-wide_256e8k deep_256e8k}"

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

launch_one() {
  local integration_variant="$1"
  local integration_tag="$2"
  local denom=$((NUM_NODES * GPUS * MICRO_BSZ))
  if (( GLOBAL_BATCH_SIZE_SEQ % denom != 0 )); then
    echo "Invalid smoke batch settings: global_batch_size_seq=${GLOBAL_BATCH_SIZE_SEQ} is not divisible by nodes*gpus*micro_bsz=${denom}" >&2
    exit 1
  fi

  local batch_tag=b768k
  local name="${RUN_PREFIX}-${integration_tag}-${LR_TAG}-${SWEEP_SUFFIX}"
  local log_path="${LOG_DIR}/${name}.log"
  local systems_tag="${batch_tag}-gpu${GPUS}-ep${EP_DIM}mb${MICRO_BSZ}"

  local cmd=(
    uv run --extra dev --extra beaker python -m olmo_core.launch.beaker
    --allow-dirty
    --name="${name}"
    --gpus "${GPUS}"
    "${common_beaker_args[@]}"
    --
    "${PYTHON_BIN}" "${SCRIPT}"
    --model-size=1p2b
    --integration-config="${integration_variant}"
    --compile
    --save-folder="${CHECKPOINT_ROOT}/${name}"
    --name="${name}"
    --data-root=s3://ai2-llm
    --lr="${LR}"
    --chinchilla-multiple="${CHINCHILLA_MULTIPLE}"
    --global-batch-size-seq="${GLOBAL_BATCH_SIZE_SEQ}"
    --num-nodes="${NUM_NODES}"
    --gpus-per-node="${GPUS}"
    --micro-batch-size="${MICRO_BSZ}"
    --ep-dim="${EP_DIM}"
    --eval-interval="${EVAL_INTERVAL}"
    --save-interval=999999999
    --ephemeral-save-interval="${EPHEMERAL_SAVE_INTERVAL}"
    --no-pre-train-checkpoint
    --tag="${integration_tag}-1p2b-cx8-smoke-${LR_TAG}-${SWEEP_SUFFIX}"
    --wandb-tag=exp_integration
    --wandb-tag="${integration_tag}"
    --wandb-tag=1p2b
    --wandb-tag=cx8-smoke
    --wandb-tag="${batch_tag}"
    --wandb-tag="${LR_TAG}"
    --wandb-tag="${systems_tag}"
    --wandb-tag=compile-on
    --wandb-tag=baseline-best-observed
    --wandb-tag=ep8-smoke
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

for integration_variant in ${INTEGRATION_VARIANTS}; do
  integration_tag="$(integration_tag_for "${integration_variant}")"
  launch_one "${integration_variant}" "${integration_tag}"
done
