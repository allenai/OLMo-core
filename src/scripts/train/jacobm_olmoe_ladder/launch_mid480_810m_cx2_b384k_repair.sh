#!/usr/bin/env bash
set -euo pipefail

# Repaired Cx2 baseline sweeps for larger model sizes.
#
# Uses the smoother Cx2 optimizer batch:
# 393,216 tokens = 48 sequences at sequence length 8192.
#
# 1.2B Cx2 is intentionally not launched here.

SCRIPT="src/scripts/train/jacobm_olmoe_ladder/tiny_275m.py"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3}"
LOG_DIR="${LOG_DIR:-/tmp/olmoe3-mid480-810m-cx2-b384k-repair-launch-logs}"
JOB_CREATED_TIMEOUT_SECONDS="${JOB_CREATED_TIMEOUT_SECONDS:-240}"
NUM_NODES=1
EP_DIM=1
GLOBAL_BATCH_SIZE_SEQ=48
CHINCHILLA_MULTIPLE=2
SWEEP_SUFFIX="${SWEEP_SUFFIX:-r1}"
EPHEMERAL_SAVE_INTERVAL="${EPHEMERAL_SAVE_INTERVAL:-500}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2000}"
LAUNCH_MID_480M="${LAUNCH_MID_480M:-1}"
LAUNCH_810M="${LAUNCH_810M:-1}"

MID_480M_LR_SPECS="${MID_480M_LR_SPECS:-4.5e-4:lr4.5e-4 9e-4:lr9e-4 1.8e-3:lr1.8e-3}"
M810_LR_SPECS="${M810_LR_SPECS:-2.8e-4:lr2.8e-4 5.6e-4:lr5.6e-4 1.12e-3:lr1.12e-3}"

if [[ "${LAUNCH_MID_480M}" != "1" && "${LAUNCH_810M}" != "1" ]]; then
  echo "Nothing to launch: set LAUNCH_MID_480M=1 and/or LAUNCH_810M=1."
  exit 1
fi

mkdir -p "${LOG_DIR}"

wait_for_job_created() {
  local name="$1"
  local log_path="$2"
  local pid="$3"
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

launch_one() {
  local model_size="$1"
  local run_prefix="$2"
  local gpus="$3"
  local micro_bsz="$4"
  local lr="$5"
  local lr_tag="$6"
  local perf_tag="gpu${gpus}-ep${EP_DIM}mb${micro_bsz}"
  local name="${run_prefix}-cx2-b384k-${perf_tag}-${lr_tag}-${SWEEP_SUFFIX}"
  local log_path="${LOG_DIR}/${name}.log"
  local common_beaker_args=(
    --cluster ai2/titan
    --nodes "${NUM_NODES}"
    --gpus "${gpus}"
    --weka oe-training-default
    --beaker-image tianhuat/olmo-core-torch211-2404-cu128
    --workspace ai2/OLMo-3-moe-experiments
    --budget ai2/oe-other
    --priority urgent
    --env OLMO_SYMM_VDEV2D_AUTO_BUILD=1
    --env-secret AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY WANDB_API_KEY=jacobm_WANDB_API_KEY
  )

  local cmd=(
    uv run --extra dev --extra beaker python -m olmo_core.launch.beaker
    --allow-dirty
    --name="${name}"
    "${common_beaker_args[@]}"
    --
    python "${SCRIPT}"
    --model-size="${model_size}"
    --save-folder="${CHECKPOINT_ROOT}/${name}"
    --name="${name}"
    --data-root=s3://ai2-llm
    --lr="${lr}"
    --chinchilla-multiple="${CHINCHILLA_MULTIPLE}"
    --global-batch-size-seq="${GLOBAL_BATCH_SIZE_SEQ}"
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
    --tag="${lr_tag}-${run_prefix}-cx2-b384k-${perf_tag}-${SWEEP_SUFFIX}"
    --wandb-tag="${model_size}"
    --wandb-tag=cx2
    --wandb-tag="${lr_tag}"
    --wandb-tag=b384k-cx2
    --wandb-tag="${perf_tag}"
    --wandb-tag=baseline
  )

  echo "Launching ${name}..."
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'

  "${cmd[@]}" >"${log_path}" 2>&1 &
  local pid=$!
  wait_for_job_created "${name}" "${log_path}" "${pid}"
}

if [[ "${LAUNCH_MID_480M}" == "1" ]]; then
  for lr_spec in ${MID_480M_LR_SPECS}; do
    launch_one mid_480m m480 4 4 "${lr_spec%%:*}" "${lr_spec##*:}"
  done
fi

if [[ "${LAUNCH_810M}" == "1" ]]; then
  for lr_spec in ${M810_LR_SPECS}; do
    launch_one 810m olmoe3-moe-a0-810m 8 2 "${lr_spec%%:*}" "${lr_spec##*:}"
  done
fi
