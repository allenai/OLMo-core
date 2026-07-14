#!/usr/bin/env bash
set -euo pipefail

SCRIPT="src/scripts/train/jacobm_olmoe_ladder/experiments/midtraining/integration_midtraining_ladder.py"
RUN_PREFIX="${RUN_PREFIX:-mt-275m}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/midtraining}"
PRETRAIN_CHECKPOINT_ROOT="${PRETRAIN_CHECKPOINT_ROOT:-/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3}"
LOG_DIR="${LOG_DIR:-/tmp/olmoe3-midtraining-275m-integration-launch-logs}"
JOB_CREATED_TIMEOUT_SECONDS="${JOB_CREATED_TIMEOUT_SECONDS:-240}"
CLUSTER="${CLUSTER:-ai2/titan}"
BEAKER_IMAGE="${BEAKER_IMAGE:-tianhuat/olmo-core-torch211-2404-cu128}"
WORKSPACE="${WORKSPACE:-ai2/OLMo-3-moe-experiments}"
BUDGET="${BUDGET:-ai2/oe-other}"
PRIORITY="${PRIORITY:-urgent}"
PREEMPTIBLE="${PREEMPTIBLE:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SWEEP_SUFFIX="${SWEEP_SUFFIX:-r2}"
MIDTRAIN_MAX_TOKENS="${MIDTRAIN_MAX_TOKENS:-100000000000}"
EPHEMERAL_SAVE_INTERVAL="${EPHEMERAL_SAVE_INTERVAL:-500}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2000}"
MODEL_SIZE="${MODEL_SIZE:-275m}"
GLOBAL_BATCH_SIZE_SEQ="${GLOBAL_BATCH_SIZE_SEQ:-128}"
GPUS="${GPUS:-4}"
NUM_NODES="${NUM_NODES:-1}"
MICRO_BSZ="${MICRO_BSZ:-8}"
EP_DIM="${EP_DIM:-1}"
INTEGRATION_VARIANTS="${INTEGRATION_VARIANTS:-wide_256e8k deep_256e8k}"
DATA_MULTIPLES="${DATA_MULTIPLES:-1 2 4 8}"

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

integration_tag_for() {
  case "$1" in
    wide_256e8k) echo intw256e8k ;;
    deep_256e8k) echo intd256e8k ;;
    *) echo "Unknown integration variant: $1" >&2; return 1 ;;
  esac
}

lr_for_cx() {
  case "$1" in
    1) echo 2e-4 ;;
    2) echo 1.8e-4 ;;
    4) echo 1.5e-4 ;;
    8) echo 1.6e-4 ;;
    *) echo "Unknown Chinchilla multiple: $1" >&2; return 1 ;;
  esac
}

lr_tag_for() {
  case "$1" in
    1.5e-4) echo lr1p5e-4 ;;
    1.6e-4) echo lr1p6e-4 ;;
    1.8e-4) echo lr1p8e-4 ;;
    2e-4) echo lr2e-4 ;;
    *) echo "lr${1}" | tr . p ;;
  esac
}

source_rel_for() {
  local integration_tag="$1"
  local cx="$2"
  case "${integration_tag}:${cx}" in
    intw256e8k:1) echo integration/int-275m-cx1-intw256e8k-lr1.6e-3-r1/step15499 ;;
    intw256e8k:2) echo integration/int-275m-cx2-intw256e8k-lr1.6e-3-r1/step20665 ;;
    intw256e8k:4) echo integration/int-275m-cx4-intw256e8k-lr8e-4-r1/step30997 ;;
    intw256e8k:8) echo integration/int-275m-cx8-intw256e8k-lr8e-4-r1/step41329 ;;
    intd256e8k:1) echo integration/int-275m-cx1-intd256e8k-lr1.6e-3-r1/step15130 ;;
    intd256e8k:2) echo integration/int-275m-cx2-intd256e8k-lr1.6e-3-r1/step20173 ;;
    intd256e8k:4) echo integration/int-275m-cx4-intd256e8k-lr1.6e-3-r1/step30259 ;;
    intd256e8k:8) echo integration/int-275m-cx8-intd256e8k-lr1.6e-3-r1/step40345 ;;
    *) echo "Unknown source checkpoint for ${integration_tag} Cx${cx}" >&2; return 1 ;;
  esac
}

launch_one() {
  local integration_variant="$1"
  local integration_tag="$2"
  local cx="$3"
  local lr
  lr="$(lr_for_cx "${cx}")"
  local lr_tag
  lr_tag="$(lr_tag_for "${lr}")"
  local source_rel
  source_rel="$(source_rel_for "${integration_tag}" "${cx}")"
  local load_path="${PRETRAIN_CHECKPOINT_ROOT}/${source_rel}"
  local batch_tag="b$((GLOBAL_BATCH_SIZE_SEQ * 8192 / 1024))k"
  local name="${RUN_PREFIX}-${integration_tag}-cx${cx}-${lr_tag}-${SWEEP_SUFFIX}"
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
    --integration-config="${integration_variant}"
    --compile
    --save-folder="${CHECKPOINT_ROOT}/${name}"
    --name="${name}"
    --data-root=s3://ai2-llm
    --load-path="${load_path}"
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
    --tag="midtrain-${integration_tag}-cx${cx}-${lr_tag}-${SWEEP_SUFFIX}"
    --wandb-tag=exp_midtraining
    --wandb-tag=exp_integration_midtraining
    --wandb-tag="${integration_tag}"
    --wandb-tag="${MODEL_SIZE}"
    --wandb-tag="cx${cx}"
    --wandb-tag="${lr_tag}"
    --wandb-tag="${systems_tag}"
    --wandb-tag=compile-on
    --wandb-tag=weight-only-load
    --wandb-tag=fresh-optimizer
    --wandb-tag=titan
  )

  echo "Launching ${name} from ${load_path}..."
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
    launch_one "${integration_variant}" "${integration_tag}" "${cx}"
  done
done
