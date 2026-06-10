#!/usr/bin/env bash
set -euo pipefail

SCRIPT="src/scripts/train/jacobm_olmoe_ladder/tiny_275m.py"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3}"
LOG_DIR="${LOG_DIR:-/tmp/olmoe3-moe-a0-mid-480m-cx4-cx8-followup-launch-logs}"
JOB_CREATED_TIMEOUT_SECONDS="${JOB_CREATED_TIMEOUT_SECONDS:-240}"
EPHEMERAL_SAVE_INTERVAL="${EPHEMERAL_SAVE_INTERVAL:-500}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2000}"
SWEEP_SUFFIX="${SWEEP_SUFFIX:-r2}"

mkdir -p "${LOG_DIR}"

launch_one() {
  local cx="$1"
  local batch_tag="$2"
  local global_batch_size_seq="$3"
  local gpus="$4"
  local micro_bsz="$5"
  local lr="$6"
  local lr_tag="$7"

  local num_nodes=1
  local ep_dim=1
  local perf_tag="gpu${gpus}-ep${ep_dim}mb${micro_bsz}"
  local name="m480-${cx}-${batch_tag}-${perf_tag}-${lr_tag}-${SWEEP_SUFFIX}"
  local log_path="${LOG_DIR}/${name}.log"
  local chinchilla_multiple="${cx#cx}"
  local common_beaker_args=(
    --cluster ai2/titan
    --nodes "${num_nodes}"
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
    --model-size=mid_480m
    --save-folder="${CHECKPOINT_ROOT}/${name}"
    --name="${name}"
    --data-root=s3://ai2-llm
    --lr="${lr}"
    --chinchilla-multiple="${chinchilla_multiple}"
    --global-batch-size-seq="${global_batch_size_seq}"
    --num-nodes="${num_nodes}"
    --gpus-per-node="${gpus}"
    --micro-batch-size="${micro_bsz}"
    --ep-dim="${ep_dim}"
    --ladder-evals
    --eval-task-set=fast
    --eval-interval="${EVAL_INTERVAL}"
    --save-interval=999999999
    --ephemeral-save-interval="${EPHEMERAL_SAVE_INTERVAL}"
    --no-pre-train-checkpoint
    --tag="${lr_tag}-m480-${cx}-${batch_tag}-${perf_tag}-${SWEEP_SUFFIX}"
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

# Wider-than-adjacent sentinels to establish missing sides of the U-curves.
launch_one cx4 b512k 64 4 8 1e-4 lr1e-4
launch_one cx8 b768k 96 8 4 3.2e-3 lr3.2e-3
