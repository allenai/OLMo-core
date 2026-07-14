#!/usr/bin/env bash
set -euo pipefail

SCRIPT="src/scripts/train/jacobm_olmoe_ladder/moe_a0_ladder.py"
RUN_PREFIX="se-275m"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/shared_expert}"
LOG_DIR="${LOG_DIR:-/tmp/olmoe3-shared-expert-275m-ladder-launch-logs}"
JOB_CREATED_TIMEOUT_SECONDS="${JOB_CREATED_TIMEOUT_SECONDS:-240}"
NUM_NODES="${NUM_NODES:-1}"
EP_DIM=1
CLUSTER="${CLUSTER:-ai2/titan}"
BEAKER_IMAGE="${BEAKER_IMAGE:-tianhuat/olmo-core-torch211-2404-cu128}"
WORKSPACE="${WORKSPACE:-ai2/OLMo-3-moe-experiments}"
BUDGET="${BUDGET:-ai2/oe-other}"
PRIORITY="${PRIORITY:-urgent}"
PREEMPTIBLE="${PREEMPTIBLE:-0}"
NO_PYTHON="${NO_PYTHON:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHONPATH_ENV="${PYTHONPATH_ENV:-}"
SWEEP_SUFFIX="${SWEEP_SUFFIX:-r1}"
EPHEMERAL_SAVE_INTERVAL="${EPHEMERAL_SAVE_INTERVAL:-500}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2000}"
SHARED_EXPERT_CONFIG="${SHARED_EXPERT_CONFIG:-no_shared_matched_active}"
SHARED_EXPERT_TAG="${SHARED_EXPERT_TAG:-se0m9}"
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

launch_one() {
  local cx="$1"
  local batch_tag="$2"
  local global_batch_size_seq="$3"
  local gpus="$4"
  local micro_bsz="$5"
  local lr="$6"
  local lr_tag="$7"
  local name="${RUN_PREFIX}-cx${cx}-${SHARED_EXPERT_TAG}-${lr_tag}-${SWEEP_SUFFIX}"
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
    --shared-expert-config="${SHARED_EXPERT_CONFIG}"
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
    --tag="${SHARED_EXPERT_TAG}-cx${cx}-${lr_tag}-${SWEEP_SUFFIX}"
    --wandb-tag=exp_shared_expert
    --wandb-tag="${SHARED_EXPERT_TAG}"
    --wandb-tag=275m
    --wandb-tag="cx${cx}"
    --wandb-tag="${lr_tag}"
    --wandb-tag="${systems_tag}"
    --wandb-tag=baseline-centered
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

for cx in ${CX_LIST}; do
  case "${cx}" in
    1)
      for lr_spec in ${CX1_LR_SPECS}; do
        launch_one 1 b256k 32 2 8 "${lr_spec%%:*}" "${lr_spec##*:}"
      done
      ;;
    2)
      for lr_spec in ${CX2_LR_SPECS}; do
        launch_one 2 b384k 48 2 8 "${lr_spec%%:*}" "${lr_spec##*:}"
      done
      ;;
    4)
      for lr_spec in ${CX4_LR_SPECS}; do
        launch_one 4 b512k 64 2 8 "${lr_spec%%:*}" "${lr_spec##*:}"
      done
      ;;
    8)
      for lr_spec in ${CX8_LR_SPECS}; do
        launch_one 8 b768k 96 4 8 "${lr_spec%%:*}" "${lr_spec##*:}"
      done
      ;;
    *)
      echo "Unsupported Cx for this launcher: ${cx}" >&2
      exit 1
      ;;
  esac
done
