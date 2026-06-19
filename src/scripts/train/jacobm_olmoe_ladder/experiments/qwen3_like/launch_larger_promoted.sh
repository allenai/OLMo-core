#!/usr/bin/env bash
set -euo pipefail

SCRIPT="src/scripts/train/jacobm_olmoe_ladder/experiments/qwen3_like/qwen3_like_ladder.py"
RUN_PREFIX="${RUN_PREFIX:-q3}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/qwen3_like}"
LOG_DIR="${LOG_DIR:-/tmp/olmoe3-qwen3-like-larger-promoted-launch-logs}"
JOB_CREATED_TIMEOUT_SECONDS="${JOB_CREATED_TIMEOUT_SECONDS:-240}"
NUM_NODES="${NUM_NODES:-1}"
EP_DIM=1
CLUSTER="${CLUSTER:-ai2/holmes}"
BEAKER_IMAGE="${BEAKER_IMAGE:-tianhuat/olmo-core-torch212-2404-cu130}"
WORKSPACE="${WORKSPACE:-ai2/holmes-testing}"
BUDGET="${BUDGET:-ai2/oe-other}"
PRIORITY="${PRIORITY:-low}"
PREEMPTIBLE="${PREEMPTIBLE:-1}"
NO_PYTHON="${NO_PYTHON:-1}"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
PYTHONPATH_ENV="${PYTHONPATH_ENV:-/gantry-runtime/src:/workspace/OLMo-core/src}"
SWEEP_SUFFIX="${SWEEP_SUFFIX:-r1}"
EPHEMERAL_SAVE_INTERVAL="${EPHEMERAL_SAVE_INTERVAL:-500}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2000}"
QWEN3_LIKE_VARIANTS="${QWEN3_LIKE_VARIANTS:-active_matched true_3d_depth_matched}"
MODEL_SIZES="${MODEL_SIZES:-480m 810m}"

mkdir -p "${LOG_DIR}"

common_beaker_args=(
  --cluster "${CLUSTER}"
  --nodes "${NUM_NODES}"
  --weka oe-training-default
  --shared-filesystem
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
if [[ "${NO_PYTHON}" == "1" ]]; then
  common_beaker_args+=(--no-python)
fi
if [[ -n "${PYTHONPATH_ENV}" ]]; then
  common_beaker_args+=(--env "PYTHONPATH=${PYTHONPATH_ENV}")
fi

qwen_tag_for() {
  case "$1" in
    active_matched) echo q3am128e8k ;;
    true_3d_depth_matched) echo q3td128e8k ;;
    *) echo "Unknown Qwen3-like variant: $1" >&2; return 1 ;;
  esac
}

launch_one() {
  local model_size="$1"
  local qwen_variant="$2"
  local qwen_tag="$3"
  local cx="$4"
  local batch_tag="$5"
  local global_batch_size_seq="$6"
  local gpus="$7"
  local micro_bsz="$8"
  local lr="$9"
  local lr_tag="${10}"
  local denom=$((NUM_NODES * gpus * micro_bsz))
  if (( global_batch_size_seq % denom != 0 )); then
    echo "Invalid batch settings for ${model_size} ${qwen_variant} Cx${cx}: global_batch_size_seq=${global_batch_size_seq} is not divisible by nodes*gpus*micro_bsz=${denom}" >&2
    exit 1
  fi

  local name="${RUN_PREFIX}-${model_size}-cx${cx}-${qwen_tag}-${lr_tag}-${SWEEP_SUFFIX}"
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
    --qwen3-like="${qwen_variant}"
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
    --tag="${qwen_tag}-${model_size}-cx${cx}-${lr_tag}-${SWEEP_SUFFIX}"
    --wandb-tag=exp_qwen3_like
    --wandb-tag="${qwen_tag}"
    --wandb-tag="${model_size}"
    --wandb-tag="cx${cx}"
    --wandb-tag="${batch_tag}"
    --wandb-tag="${lr_tag}"
    --wandb-tag="${systems_tag}"
    --wandb-tag=compile-on
    --wandb-tag=baseline-best-observed
    --wandb-tag=promoted-single-point
    --wandb-tag=holmes
    --wandb-tag=preemptible
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

launch_model() {
  local model_size="$1"
  local qwen_variant="$2"
  local qwen_tag
  qwen_tag="$(qwen_tag_for "${qwen_variant}")"

  case "${model_size}" in
    480m)
      launch_one 480m "${qwen_variant}" "${qwen_tag}" 1 b256k 32 4 8 1.2e-3 lr1.2e-3
      launch_one 480m "${qwen_variant}" "${qwen_tag}" 2 b384k 48 4 4 9e-4 lr9e-4
      launch_one 480m "${qwen_variant}" "${qwen_tag}" 4 b512k 64 4 8 8e-4 lr8e-4
      launch_one 480m "${qwen_variant}" "${qwen_tag}" 8 b768k 96 8 4 8e-4 lr8e-4
      ;;
    810m)
      launch_one 810m "${qwen_variant}" "${qwen_tag}" 1 b256k 32 8 4 6e-4 lr6e-4
      launch_one 810m "${qwen_variant}" "${qwen_tag}" 2 b384k 48 8 2 5.6e-4 lr5.6e-4
      launch_one 810m "${qwen_variant}" "${qwen_tag}" 4 b512k 64 8 4 4e-4 lr4e-4
      launch_one 810m "${qwen_variant}" "${qwen_tag}" 8 b768k 96 8 4 4e-4 lr4e-4
      ;;
    *)
      echo "Unsupported promoted Qwen3-like model size: ${model_size}" >&2
      exit 1
      ;;
  esac
}

for model_size in ${MODEL_SIZES}; do
  for qwen_variant in ${QWEN3_LIKE_VARIANTS}; do
    launch_model "${model_size}" "${qwen_variant}"
  done
done
