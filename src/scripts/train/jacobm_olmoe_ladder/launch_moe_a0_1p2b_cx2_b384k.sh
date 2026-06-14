#!/usr/bin/env bash
set -euo pipefail

# 1.2B baseline Cx2 repaired-batch sweep.
# Names are semantic/resume-stable: systems settings are W&B tags/config only.

SCRIPT="src/scripts/train/jacobm_olmoe_ladder/moe_a0_ladder.py"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3}"
LOG_DIR="${LOG_DIR:-/tmp/olmoe3-1p2b-cx2-b384k-launch-logs}"
JOB_CREATED_TIMEOUT_SECONDS="${JOB_CREATED_TIMEOUT_SECONDS:-240}"
NUM_NODES="${NUM_NODES:-1}"
GPUS="${GPUS:-8}"
EP_DIM="${EP_DIM:-1}"
MICRO_BSZ="${MICRO_BSZ:-2}"
GLOBAL_BATCH_SIZE_SEQ=48
CHINCHILLA_MULTIPLE=2
SWEEP_SUFFIX="${SWEEP_SUFFIX:-r1}"
EPHEMERAL_SAVE_INTERVAL="${EPHEMERAL_SAVE_INTERVAL:-500}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2000}"
LR_SPECS="${LR_SPECS:-1.5e-4:lr1.5e-4 3e-4:lr3e-4 6e-4:lr6e-4}"

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
  local lr="$1"
  local lr_tag="$2"
  local name="olmoe3-moe-a0-1p2b-cx2-b384k-${lr_tag}-${SWEEP_SUFFIX}"
  local log_path="${LOG_DIR}/${name}.log"
  local common_beaker_args=(
    --cluster ai2/titan
    --nodes "${NUM_NODES}"
    --gpus "${GPUS}"
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
    --model-size=1p2b
    --save-folder="${CHECKPOINT_ROOT}/${name}"
    --name="${name}"
    --data-root=s3://ai2-llm
    --lr="${lr}"
    --chinchilla-multiple="${CHINCHILLA_MULTIPLE}"
    --global-batch-size-seq="${GLOBAL_BATCH_SIZE_SEQ}"
    --num-nodes="${NUM_NODES}"
    --gpus-per-node="${GPUS}"
    --micro-batch-size="${MICRO_BSZ}"
    --ep-dim="${EP_DIM}"
    --ladder-evals
    --eval-task-set=fast
    --eval-interval="${EVAL_INTERVAL}"
    --save-interval=999999999
    --ephemeral-save-interval="${EPHEMERAL_SAVE_INTERVAL}"
    --no-pre-train-checkpoint
    --tag="1p2b-cx2-b384k-${lr_tag}-${SWEEP_SUFFIX}"
    --wandb-tag=baseline
    --wandb-tag=1p2b
    --wandb-tag=cx2
    --wandb-tag=b384k-cx2
    --wandb-tag="${lr_tag}"
    --wandb-tag=nodes${NUM_NODES}
    --wandb-tag=gpu${GPUS}
    --wandb-tag=ep${EP_DIM}
    --wandb-tag=mb${MICRO_BSZ}
    --wandb-tag=compute-efficient
  )

  echo "Launching ${name}..."
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '
'

  "${cmd[@]}" >"${log_path}" 2>&1 &
  local pid=$!
  wait_for_job_created "${name}" "${log_path}" "${pid}"
}

for lr_spec in ${LR_SPECS}; do
  launch_one "${lr_spec%%:*}" "${lr_spec##*:}"
done
