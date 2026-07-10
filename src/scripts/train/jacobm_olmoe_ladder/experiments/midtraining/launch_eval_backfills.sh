#!/usr/bin/env bash
set -euo pipefail

SCRIPT="${SCRIPT:-src/scripts/train/jacobm_olmoe_ladder/experiments/midtraining/midtraining_ladder.py}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/midtraining}"
EVAL_ROOT="${EVAL_ROOT:-/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/midtraining/eval-backfills}"
LOG_DIR="${LOG_DIR:-/tmp/olmoe3-midtraining-eval-backfill-launch-logs}"
JOB_CREATED_TIMEOUT_SECONDS="${JOB_CREATED_TIMEOUT_SECONDS:-240}"
CLUSTER="${CLUSTER:-ai2/titan}"
BEAKER_IMAGE="${BEAKER_IMAGE:-tianhuat/olmo-core-torch211-2404-cu128}"
WORKSPACE="${WORKSPACE:-ai2/OLMo-3-moe-experiments}"
BUDGET="${BUDGET-ai2/oe-other}"
PRIORITY="${PRIORITY:-urgent}"
PREEMPTIBLE="${PREEMPTIBLE:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_SIZE="${MODEL_SIZE:-275m}"
GPUS="${GPUS:-2}"
NUM_NODES="${NUM_NODES:-1}"
DATA_ROOT="${DATA_ROOT:-s3://ai2-llm}"
EVAL_TASK_SET="${EVAL_TASK_SET:-fast}"
MIDTRAIN_MAX_TOKENS="${MIDTRAIN_MAX_TOKENS:-100000000000}"
read -r -a EXTRA_SCRIPT_ARGS <<< "${EXTRA_SCRIPT_ARGS:-}"

# Defaults cover the six currently finished 275M midtraining LR-grid runs.
# The two 1.6e-3 runs should be added after their final checkpoints exist.
TARGETS=(
  "mt-275m-baseline-cx1-lr2e-4-r1:step95368"
  "mt-275m-baseline-cx1-lr4e-4-r1:step95368"
  "mt-275m-baseline-cx1-lr8e-4-r1:step95368"
  "mt-275m-baseline-cx8-lr2e-4-r1:step95368"
  "mt-275m-baseline-cx8-lr4e-4-r1:step95368"
  "mt-275m-baseline-cx8-lr8e-4-r1:step95368"
)

if [[ $# -gt 0 ]]; then
  TARGETS=("$@")
fi

mkdir -p "${LOG_DIR}"

common_beaker_args=(
  --cluster "${CLUSTER}"
  --nodes "${NUM_NODES}"
  --gpus "${GPUS}"
  --weka oe-training-default
  --beaker-image "${BEAKER_IMAGE}"
  --workspace "${WORKSPACE}"
  --priority "${PRIORITY}"
  --env OLMO_SYMM_VDEV2D_AUTO_BUILD=1
  --env-secret AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY WANDB_API_KEY=jacobm_WANDB_API_KEY
)
if [[ -n "${BUDGET}" ]]; then
  common_beaker_args+=(--budget "${BUDGET}")
fi
if [[ "${PREEMPTIBLE}" == "1" ]]; then
  common_beaker_args+=(--preemptible)
fi

launch_one() {
  local target="$1"
  local source_name="${target%%:*}"
  local step_name="${target#*:}"
  if [[ "${source_name}" == "${step_name}" ]]; then
    step_name="step95368"
  fi
  local source_checkpoint="${CHECKPOINT_ROOT}/${source_name}/${step_name}"
  local eval_name="mt-eval-${source_name#mt-}"
  local save_folder="${EVAL_ROOT}/${eval_name}"
  local log_path="${LOG_DIR}/${eval_name}.log"

  if [[ ! -d "${source_checkpoint}" ]]; then
    echo "Missing checkpoint directory: ${source_checkpoint}" >&2
    return 1
  fi

  local cmd=(
    uv run --extra dev --extra beaker python -m olmo_core.launch.beaker
    --allow-dirty
    --name="${eval_name}"
    "${common_beaker_args[@]}"
    --
    "${PYTHON_BIN}" "${SCRIPT}"
    --model-size="${MODEL_SIZE}"
    --save-folder="${save_folder}"
    --name="${eval_name}"
    --data-root="${DATA_ROOT}"
    "${EXTRA_SCRIPT_ARGS[@]}"
    --eval-checkpoints "${source_checkpoint}"
    --midtrain-max-tokens="${MIDTRAIN_MAX_TOKENS}"
    --eval-task-set="${EVAL_TASK_SET}"
    --tag=midtraining-eval-backfill
    --wandb-tag=eval-backfill
    --wandb-tag=midtraining
    --wandb-tag=exp_midtraining_eval_backfill
    --wandb-tag="source-${source_name}"
  )

  echo "Launching ${eval_name} for ${source_checkpoint}..."
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
      echo "Detached local launcher for ${eval_name}; Beaker job continues."
      return 0
    fi

    if ! kill -0 "${pid}" 2>/dev/null; then
      cat "${log_path}"
      wait "${pid}"
      return $?
    fi

    sleep 2
  done

  echo "Timed out waiting for Beaker job creation for ${eval_name}; log follows:"
  cat "${log_path}"
  kill "${pid}" 2>/dev/null || true
  wait "${pid}" 2>/dev/null || true
  return 1
}

for target in "${TARGETS[@]}"; do
  launch_one "${target}"
done
