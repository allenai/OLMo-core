#!/usr/bin/env bash
set -euo pipefail

SCRIPT="src/scripts/train/jacobm_olmoe_ladder/experiments/midtraining/integration_midtraining_ladder.py"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/midtraining}"
PRETRAIN_CHECKPOINT_ROOT="${PRETRAIN_CHECKPOINT_ROOT:-/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3}"
LOG_DIR="${LOG_DIR:-/tmp/olmoe3-midtraining-275m-top16-launch-logs}"
JOB_CREATED_TIMEOUT_SECONDS="${JOB_CREATED_TIMEOUT_SECONDS:-240}"
CLUSTER="${CLUSTER:-ai2/titan}"
BEAKER_IMAGE="${BEAKER_IMAGE:-tianhuat/olmo-core-torch211-2404-cu128}"
WORKSPACE="${WORKSPACE:-ai2/OLMo-3-moe-experiments}"
BUDGET="${BUDGET:-ai2/oe-other}"
PRIORITY="${PRIORITY:-urgent}"
PREEMPTIBLE="${PREEMPTIBLE:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL_SIZE="${MODEL_SIZE:-275m}"
INTEGRATION_VARIANT="${INTEGRATION_VARIANT:-wide_256e16k}"
INTEGRATION_TAG="${INTEGRATION_TAG:-intw256e16k}"
SOURCE_REL="${SOURCE_REL:-integration/int-275m-cx4-intw256e16k-lr8e-4-r1/step61993}"
LR="${LR:-8e-5}"
LR_TAG="${LR_TAG:-lr8e-5}"
NAME="${NAME:-mt-275m-intw256e16k-cx4-lr8e-5-r1}"
MIDTRAIN_MAX_TOKENS="${MIDTRAIN_MAX_TOKENS:-100000000000}"
GLOBAL_BATCH_SIZE_SEQ="${GLOBAL_BATCH_SIZE_SEQ:-128}"
GPUS="${GPUS:-8}"
MICRO_BSZ="${MICRO_BSZ:-4}"
EP_DIM="${EP_DIM:-1}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2000}"
EPHEMERAL_SAVE_INTERVAL="${EPHEMERAL_SAVE_INTERVAL:-500}"

denom=$((GPUS * MICRO_BSZ))
if (( GLOBAL_BATCH_SIZE_SEQ % denom != 0 )); then
  echo "Invalid batch settings: GLOBAL_BATCH_SIZE_SEQ=${GLOBAL_BATCH_SIZE_SEQ} is not divisible by GPUS*MICRO_BSZ=${denom}" >&2
  exit 1
fi

load_path="${PRETRAIN_CHECKPOINT_ROOT}/${SOURCE_REL}"
if [[ ! -d "${load_path}" ]]; then
  echo "Missing source checkpoint: ${load_path}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"
batch_tag="b$((GLOBAL_BATCH_SIZE_SEQ * 8192 / 1024))k"
systems_tag="${batch_tag}-gpu${GPUS}-ep${EP_DIM}mb${MICRO_BSZ}"
log_path="${LOG_DIR}/${NAME}.log"

cmd=(
  uv run --extra dev --extra beaker python -m olmo_core.launch.beaker
  --allow-dirty
  --name="${NAME}"
  --cluster "${CLUSTER}"
  --nodes 1
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
  cmd+=(--preemptible)
fi
cmd+=(
  --
  "${PYTHON_BIN}" "${SCRIPT}"
  --model-size="${MODEL_SIZE}"
  --integration-config="${INTEGRATION_VARIANT}"
  --compile
  --save-folder="${CHECKPOINT_ROOT}/${NAME}"
  --name="${NAME}"
  --data-root=s3://ai2-llm
  --load-path="${load_path}"
  --lr="${LR}"
  --midtrain-max-tokens="${MIDTRAIN_MAX_TOKENS}"
  --global-batch-size-seq="${GLOBAL_BATCH_SIZE_SEQ}"
  --num-nodes=1
  --gpus-per-node="${GPUS}"
  --micro-batch-size="${MICRO_BSZ}"
  --ep-dim="${EP_DIM}"
  --eval-task-set=fast
  --eval-interval="${EVAL_INTERVAL}"
  --save-interval=999999999
  --ephemeral-save-interval="${EPHEMERAL_SAVE_INTERVAL}"
  --no-pre-train-checkpoint
  --tag="midtrain-${INTEGRATION_TAG}-cx4-${LR_TAG}"
  --wandb-tag=exp_midtraining
  --wandb-tag=exp_integration_midtraining
  --wandb-tag="${INTEGRATION_TAG}"
  --wandb-tag="${MODEL_SIZE}"
  --wandb-tag=cx4
  --wandb-tag="${LR_TAG}"
  --wandb-tag="${systems_tag}"
  --wandb-tag=compile-on
  --wandb-tag=weight-only-load
  --wandb-tag=fresh-optimizer
  --wandb-tag=top16-active
  --wandb-tag=data-matched-275m-cx8
  --wandb-tag=titan
)

echo "Launching ${NAME} from ${load_path}..."
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
    echo "Detached local launcher for ${NAME}; Beaker job continues."
    exit 0
  fi

  if ! kill -0 "${pid}" 2>/dev/null; then
    cat "${log_path}"
    wait "${pid}"
    exit $?
  fi

  sleep 2
done

echo "Timed out waiting for Beaker job creation for ${NAME}; log follows:"
cat "${log_path}"
kill "${pid}" 2>/dev/null || true
wait "${pid}" 2>/dev/null || true
exit 1
