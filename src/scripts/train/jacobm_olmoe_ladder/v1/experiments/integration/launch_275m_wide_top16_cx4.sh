#!/usr/bin/env bash
set -euo pipefail

SCRIPT="src/scripts/train/jacobm_olmoe_ladder/experiments/integration/integration_ladder.py"
RUN_PREFIX="${RUN_PREFIX:-int-275m}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/integration}"
LOG_DIR="${LOG_DIR:-/tmp/olmoe3-integration-275m-top16-launch-logs}"
JOB_CREATED_TIMEOUT_SECONDS="${JOB_CREATED_TIMEOUT_SECONDS:-240}"
NUM_NODES="${NUM_NODES:-1}"
GPUS="${GPUS:-8}"
EP_DIM="${EP_DIM:-1}"
GLOBAL_BATCH_SIZE_SEQ="${GLOBAL_BATCH_SIZE_SEQ:-64}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
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
CX="${CX:-4}"
LR="${LR:-8e-4}"
LR_TAG="${LR_TAG:-lr8e-4}"
INTEGRATION_VARIANT="${INTEGRATION_VARIANT:-wide_256e16k}"
INTEGRATION_TAG="${INTEGRATION_TAG:-intw256e16k}"
TARGET_TOKENS="${TARGET_TOKENS:-32502185984}"

mkdir -p "${LOG_DIR}"

denom=$((NUM_NODES * GPUS * MICRO_BATCH_SIZE))
if (( GLOBAL_BATCH_SIZE_SEQ % denom != 0 )); then
  echo "Invalid batch settings: global_batch_size_seq=${GLOBAL_BATCH_SIZE_SEQ} is not divisible by nodes*gpus*micro_bsz=${denom}" >&2
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

name="${RUN_PREFIX}-cx${CX}-${INTEGRATION_TAG}-${LR_TAG}-${SWEEP_SUFFIX}"
log_path="${LOG_DIR}/${name}.log"
systems_tag="b512k-gpu${GPUS}-ep${EP_DIM}mb${MICRO_BATCH_SIZE}"

cmd=(
  uv run --extra dev --extra beaker python -m olmo_core.launch.beaker
  --allow-dirty
  --name="${name}"
  "${common_beaker_args[@]}"
  --
  "${PYTHON_BIN}" "${SCRIPT}"
  --model-size=275m
  --integration-config="${INTEGRATION_VARIANT}"
  --compile
  --save-folder="${CHECKPOINT_ROOT}/${name}"
  --name="${name}"
  --data-root=s3://ai2-llm
  --lr="${LR}"
  --chinchilla-multiple="${CX}"
  --max-duration-tokens="${TARGET_TOKENS}"
  --global-batch-size-seq="${GLOBAL_BATCH_SIZE_SEQ}"
  --num-nodes="${NUM_NODES}"
  --gpus-per-node="${GPUS}"
  --micro-batch-size="${MICRO_BATCH_SIZE}"
  --ep-dim="${EP_DIM}"
  --ladder-evals
  --eval-task-set=fast
  --eval-interval="${EVAL_INTERVAL}"
  --save-interval=999999999
  --ephemeral-save-interval="${EPHEMERAL_SAVE_INTERVAL}"
  --no-pre-train-checkpoint
  --tag="${INTEGRATION_TAG}-cx${CX}-${LR_TAG}-${SWEEP_SUFFIX}"
  --wandb-tag=exp_integration
  --wandb-tag="${INTEGRATION_TAG}"
  --wandb-tag=275m
  --wandb-tag="cx${CX}"
  --wandb-tag="${LR_TAG}"
  --wandb-tag="${systems_tag}"
  --wandb-tag=compile-on
  --wandb-tag=baseline-centered
  --wandb-tag=top16-active
  --wandb-tag=titan
)

echo "Launching ${name}..."
printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}" >"${log_path}" 2>&1 &
pid=$!
deadline=$((SECONDS + JOB_CREATED_TIMEOUT_SECONDS))

while (( SECONDS < deadline )); do
  if [[ -f "${log_path}" ]] && grep -q "job created" "${log_path}"; then
    sed -n '1,/job created/p' "${log_path}"
    kill "${pid}" 2>/dev/null || true
    wait "${pid}" 2>/dev/null || true
    echo "Detached local launcher for ${name}; Beaker job continues."
    exit 0
  fi

  if ! kill -0 "${pid}" 2>/dev/null; then
    cat "${log_path}"
    wait "${pid}"
    exit $?
  fi

  sleep 2
done

echo "Timed out waiting for Beaker job creation for ${name}; log follows:"
cat "${log_path}"
kill "${pid}" 2>/dev/null || true
wait "${pid}" 2>/dev/null || true
exit 1
