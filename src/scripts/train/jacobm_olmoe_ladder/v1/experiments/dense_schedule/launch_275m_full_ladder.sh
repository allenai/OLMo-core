#!/usr/bin/env bash
set -euo pipefail

SCRIPT="src/scripts/train/jacobm_olmoe_ladder/moe_a0_ladder.py"
RUN_PREFIX="${RUN_PREFIX:-ds-275m}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/dense_schedule}"
LOG_DIR="${LOG_DIR:-/tmp/olmoe3-dense-schedule-275m-ladder-launch-logs}"
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
COMPILE="${COMPILE:-0}"
SWEEP_SUFFIX="${SWEEP_SUFFIX:-r1}"
EPHEMERAL_SAVE_INTERVAL="${EPHEMERAL_SAVE_INTERVAL:-500}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2000}"
DENSE_SCHEDULES="${DENSE_SCHEDULES:-dense0_shared dense2_shared dense4_shared}"
CX_LIST="${CX_LIST:-1 2 4 8}"
CX1_LR_SPECS="${CX1_LR_SPECS:-1e-3:lr1e-3 2e-3:lr2e-3 4e-3:lr4e-3}"
CX2_LR_SPECS="${CX2_LR_SPECS:-9e-4:lr9e-4 1.8e-3:lr1.8e-3 3.6e-3:lr3.6e-3}"
CX4_LR_SPECS="${CX4_LR_SPECS:-8e-4:lr8e-4 1.6e-3:lr1.6e-3 3.2e-3:lr3.2e-3}"
CX8_LR_SPECS="${CX8_LR_SPECS:-8e-4:lr8e-4 1.6e-3:lr1.6e-3 3.2e-3:lr3.2e-3}"

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
if [[ "${NO_PYTHON}" == "1" ]]; then
  common_beaker_args+=(--no-python)
fi
if [[ -n "${PYTHONPATH_ENV}" ]]; then
  common_beaker_args+=(--env "PYTHONPATH=${PYTHONPATH_ENV}")
fi

compile_args=()
compile_tag=compile-on
if [[ "${COMPILE}" == "0" ]]; then
  compile_args+=(--no-compile)
  compile_tag=compile-off
fi

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
  local dense_schedule="$1"
  local dense_tag="$2"
  local cx="$3"
  local batch_tag="$4"
  local global_batch_size_seq="$5"
  local gpus="$6"
  local micro_bsz="$7"
  local lr="$8"
  local lr_tag="$9"
  local name="${RUN_PREFIX}-cx${cx}-${dense_tag}-${lr_tag}-${SWEEP_SUFFIX}"
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
    --dense-schedule="${dense_schedule}"
    "${compile_args[@]}"
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
    --tag="${dense_tag}-cx${cx}-${lr_tag}-${SWEEP_SUFFIX}"
    --wandb-tag=exp_dense_schedule
    --wandb-tag="${dense_tag}"
    --wandb-tag=275m
    --wandb-tag="cx${cx}"
    --wandb-tag="${lr_tag}"
    --wandb-tag="${systems_tag}"
    --wandb-tag="${compile_tag}"
    --wandb-tag=baseline-centered
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

for dense_schedule in ${DENSE_SCHEDULES}; do
  dense_tag="$(dense_tag_for "${dense_schedule}")"
  for cx in ${CX_LIST}; do
    case "${cx}" in
      1)
        for lr_spec in ${CX1_LR_SPECS}; do
          launch_one "${dense_schedule}" "${dense_tag}" 1 b256k 32 2 8 "${lr_spec%%:*}" "${lr_spec##*:}"
        done
        ;;
      2)
        for lr_spec in ${CX2_LR_SPECS}; do
          launch_one "${dense_schedule}" "${dense_tag}" 2 b384k 48 2 8 "${lr_spec%%:*}" "${lr_spec##*:}"
        done
        ;;
      4)
        for lr_spec in ${CX4_LR_SPECS}; do
          launch_one "${dense_schedule}" "${dense_tag}" 4 b512k 64 2 8 "${lr_spec%%:*}" "${lr_spec##*:}"
        done
        ;;
      8)
        for lr_spec in ${CX8_LR_SPECS}; do
          launch_one "${dense_schedule}" "${dense_tag}" 8 b768k 96 4 8 "${lr_spec%%:*}" "${lr_spec##*:}"
        done
        ;;
      *)
        echo "Unsupported Cx for this launcher: ${cx}" >&2
        exit 1
        ;;
    esac
  done
done
