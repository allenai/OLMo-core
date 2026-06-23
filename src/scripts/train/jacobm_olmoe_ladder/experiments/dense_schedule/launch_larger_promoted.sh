#!/usr/bin/env bash
set -euo pipefail

SCRIPT="src/scripts/train/jacobm_olmoe_ladder/moe_a0_ladder.py"
RUN_PREFIX="${RUN_PREFIX:-ds}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/dense_schedule}"
LOG_DIR="${LOG_DIR:-/tmp/olmoe3-dense-schedule-larger-promoted-launch-logs}"
JOB_CREATED_TIMEOUT_SECONDS="${JOB_CREATED_TIMEOUT_SECONDS:-240}"
NUM_NODES="${NUM_NODES:-1}"
EP_DIM=1
CLUSTER="${CLUSTER:-ai2/titan}"
BEAKER_IMAGE="${BEAKER_IMAGE:-tianhuat/olmo-core-torch211-2404-cu128}"
WORKSPACE="${WORKSPACE:-ai2/OLMo-3-moe-experiments}"
BUDGET="${BUDGET:-ai2/oe-other}"
PRIORITY="${PRIORITY:-urgent}"
SWEEP_SUFFIX="${SWEEP_SUFFIX:-r1}"
EPHEMERAL_SAVE_INTERVAL="${EPHEMERAL_SAVE_INTERVAL:-500}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2000}"
DENSE_SCHEDULES="${DENSE_SCHEDULES:-dense0_shared dense2_shared dense4_shared}"
MODEL_SIZES="${MODEL_SIZES:-480m 810m 1p2b}"

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

dense_tag_for() {
  case "$1" in
    dense0_shared) echo ds0-sh ;;
    dense1_shared) echo ds1-sh ;;
    dense2_shared) echo ds2-sh ;;
    dense4_shared) echo ds4-sh ;;
    *) echo "Unknown dense schedule: $1" >&2; return 1 ;;
  esac
}

launch_one() {
  local model_size="$1"
  local dense_schedule="$2"
  local dense_tag="$3"
  local cx="$4"
  local batch_tag="$5"
  local global_batch_size_seq="$6"
  local gpus="$7"
  local micro_bsz="$8"
  local lr="$9"
  local lr_tag="${10}"
  local denom=$((NUM_NODES * gpus * micro_bsz))
  if (( global_batch_size_seq % denom != 0 )); then
    echo "Invalid batch settings for ${model_size} ${dense_tag} Cx${cx}: global_batch_size_seq=${global_batch_size_seq} is not divisible by nodes*gpus*micro_bsz=${denom}" >&2
    exit 1
  fi

  local name="${RUN_PREFIX}-${model_size}-cx${cx}-${dense_tag}-${lr_tag}-${SWEEP_SUFFIX}"
  local log_path="${LOG_DIR}/${name}.log"
  local systems_tag="${batch_tag}-gpu${gpus}-ep${EP_DIM}mb${micro_bsz}"

  local cmd=(
    uv run --extra dev --extra beaker python -m olmo_core.launch.beaker
    --allow-dirty
    --name="${name}"
    --gpus "${gpus}"
    "${common_beaker_args[@]}"
    --
    python "${SCRIPT}"
    --model-size="${model_size}"
    --dense-schedule="${dense_schedule}"
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
    --tag="${dense_tag}-${model_size}-cx${cx}-${lr_tag}-${SWEEP_SUFFIX}"
    --wandb-tag=exp_dense_schedule
    --wandb-tag="${dense_tag}"
    --wandb-tag="${model_size}"
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

launch_model() {
  local model_size="$1"
  local dense_schedule="$2"
  local dense_tag
  dense_tag="$(dense_tag_for "${dense_schedule}")"

  case "${model_size}" in
    480m)
      launch_one 480m "${dense_schedule}" "${dense_tag}" 1 b256k 32 4 8 1.2e-3 lr1.2e-3
      launch_one 480m "${dense_schedule}" "${dense_tag}" 2 b384k 48 4 4 9e-4 lr9e-4
      launch_one 480m "${dense_schedule}" "${dense_tag}" 4 b512k 64 4 8 8e-4 lr8e-4
      launch_one 480m "${dense_schedule}" "${dense_tag}" 8 b768k 96 8 4 8e-4 lr8e-4
      ;;
    810m)
      launch_one 810m "${dense_schedule}" "${dense_tag}" 1 b256k 32 8 4 6e-4 lr6e-4
      launch_one 810m "${dense_schedule}" "${dense_tag}" 2 b384k 48 8 2 5.6e-4 lr5.6e-4
      launch_one 810m "${dense_schedule}" "${dense_tag}" 4 b512k 64 8 4 4e-4 lr4e-4
      launch_one 810m "${dense_schedule}" "${dense_tag}" 8 b768k 96 8 4 4e-4 lr4e-4
      ;;
    1p2b)
      launch_one 1p2b "${dense_schedule}" "${dense_tag}" 1 b256k 32 8 2 4e-4 lr4e-4
      launch_one 1p2b "${dense_schedule}" "${dense_tag}" 2 b384k 48 8 2 6e-4 lr6e-4
      launch_one 1p2b "${dense_schedule}" "${dense_tag}" 4 b512k 64 8 2 3e-4 lr3e-4
      launch_one 1p2b "${dense_schedule}" "${dense_tag}" 8 b768k 96 8 4 4e-4 lr4e-4
      ;;
    *)
      echo "Unsupported promoted dense-schedule model size: ${model_size}" >&2
      exit 1
      ;;
  esac
}

for model_size in ${MODEL_SIZES}; do
  for dense_schedule in ${DENSE_SCHEDULES}; do
    launch_model "${model_size}" "${dense_schedule}"
  done
done
