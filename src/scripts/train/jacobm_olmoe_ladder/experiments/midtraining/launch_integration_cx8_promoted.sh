#!/usr/bin/env bash
set -euo pipefail

SCRIPT="src/scripts/train/jacobm_olmoe_ladder/experiments/midtraining/integration_midtraining_ladder.py"
RUN_PREFIX="${RUN_PREFIX:-mt}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/midtraining}"
PRETRAIN_CHECKPOINT_ROOT="${PRETRAIN_CHECKPOINT_ROOT:-/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3}"
LOG_DIR="${LOG_DIR:-/tmp/olmoe3-midtraining-integration-cx8-launch-logs}"
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
MODEL_SIZES="${MODEL_SIZES:-480m 810m 1p2b}"
INTEGRATION_VARIANTS="${INTEGRATION_VARIANTS:-wide_256e8k deep_256e8k}"
EP_DIM="${EP_DIM:-1}"

mkdir -p "${LOG_DIR}"

common_beaker_args=(
  --cluster "${CLUSTER}"
  --nodes 1
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

settings_for_size() {
  case "$1" in
    480m) echo "8e-5 lr8e-5 192 4 8" ;;
    810m) echo "4e-5 lr4e-5 256 8 4" ;;
    1p2b) echo "4e-5 lr4e-5 384 8 4" ;;
    *) echo "Unknown model size: $1" >&2; return 1 ;;
  esac
}

source_rel_for() {
  local model_size="$1"
  local integration_tag="$2"
  case "${model_size}:${integration_tag}" in
    480m:intw256e8k) echo integration/int-480m-cx8-intw256e8k-lr8e-4-r1/step78042 ;;
    810m:intw256e8k) echo integration/int-810m-cx8-intw256e8k-lr4e-4-r1/step141423 ;;
    1p2b:intw256e8k) echo integration/int-1p2b-cx8-intw256e8k-lr4e-4-r2/step217870 ;;
    480m:intd256e8k) echo integration/int-480m-cx8-intd256e8k-lr8e-4-r1/step78659 ;;
    810m:intd256e8k) echo integration/int-810m-cx8-intd256e8k-lr4e-4-r1/step138619 ;;
    1p2b:intd256e8k) echo integration/int-1p2b-cx8-intd256e8k-lr4e-4-r2/step210000 ;;
    *) echo "Unknown Cx8 source checkpoint for ${model_size} ${integration_tag}" >&2; return 1 ;;
  esac
}

launch_one() {
  local model_size="$1"
  local integration_variant="$2"
  local integration_tag="$3"

  read -r lr lr_tag global_batch_size_seq gpus micro_bsz < <(settings_for_size "${model_size}")
  local denom=$((gpus * micro_bsz))
  if (( global_batch_size_seq % denom != 0 )); then
    echo "Invalid batch settings for ${model_size} ${integration_tag}: global_batch_size_seq=${global_batch_size_seq} is not divisible by gpus*micro_bsz=${denom}" >&2
    exit 1
  fi

  local source_rel
  source_rel="$(source_rel_for "${model_size}" "${integration_tag}")"
  local load_path="${PRETRAIN_CHECKPOINT_ROOT}/${source_rel}"
  if [[ ! -d "${load_path}" ]]; then
    echo "Missing source checkpoint: ${load_path}" >&2
    return 1
  fi

  local batch_tag="b$((global_batch_size_seq * 8192 / 1024))k"
  local name="${RUN_PREFIX}-${model_size}-${integration_tag}-cx8-${lr_tag}-${SWEEP_SUFFIX}"
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
    --model-size="${model_size}"
    --integration-config="${integration_variant}"
    --compile
    --save-folder="${CHECKPOINT_ROOT}/${name}"
    --name="${name}"
    --data-root=s3://ai2-llm
    --load-path="${load_path}"
    --lr="${lr}"
    --midtrain-max-tokens="${MIDTRAIN_MAX_TOKENS}"
    --global-batch-size-seq="${global_batch_size_seq}"
    --num-nodes=1
    --gpus-per-node="${gpus}"
    --micro-batch-size="${micro_bsz}"
    --ep-dim="${EP_DIM}"
    --eval-task-set=fast
    --eval-interval="${EVAL_INTERVAL}"
    --save-interval=999999999
    --ephemeral-save-interval="${EPHEMERAL_SAVE_INTERVAL}"
    --no-pre-train-checkpoint
    --tag="midtrain-${integration_tag}-${model_size}-cx8-${lr_tag}-${SWEEP_SUFFIX}"
    --wandb-tag=exp_midtraining
    --wandb-tag=exp_integration_midtraining
    --wandb-tag="${integration_tag}"
    --wandb-tag="${model_size}"
    --wandb-tag=cx8
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
  for model_size in ${MODEL_SIZES}; do
    launch_one "${model_size}" "${integration_variant}" "${integration_tag}"
  done
done
