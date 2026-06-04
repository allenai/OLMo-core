#!/usr/bin/env bash
set -euo pipefail

SCRIPT="src/scripts/train/jacobm_olmoe_ladder/tiny_275m.py"
RUN_PREFIX="olmoe3-tiny-275m"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3}"
LOG_DIR="${LOG_DIR:-/tmp/olmoe3-tiny-275m-cx8-cx16-lr-rule-launch-logs}"
JOB_CREATED_TIMEOUT_SECONDS="${JOB_CREATED_TIMEOUT_SECONDS:-240}"
NUM_NODES=1

mkdir -p "${LOG_DIR}"

launch_one() {
  local chinchilla="$1"
  local batch_tag="$2"
  local batch_seq="$3"
  local gpus="$4"
  local micro_bsz="$5"
  local lr="$6"
  local lr_tag="$7"
  local perf_tag="gpu${gpus}-ep1mb${micro_bsz}"
  local name="${RUN_PREFIX}-cx${chinchilla}-${batch_tag}-${perf_tag}-${lr_tag}"
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
    --name="${name}" \
    "${common_beaker_args[@]}" \
    -- \
    python "${SCRIPT}"
    --save-folder="${CHECKPOINT_ROOT}/${name}"
    --name="${name}"
    --data-root=s3://ai2-llm
    --lr="${lr}"
    --chinchilla-multiple="${chinchilla}"
    --global-batch-size-seq="${batch_seq}"
    --num-nodes="${NUM_NODES}"
    --gpus-per-node="${gpus}"
    --micro-batch-size="${micro_bsz}"
    --ep-dim=1
    --tag="${lr_tag}-cx${chinchilla}-${batch_tag}-${perf_tag}"
  )

  echo "Launching ${name}..."
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'

  "${cmd[@]}" >"${log_path}" 2>&1 &
  local pid=$!
  local deadline=$((SECONDS + JOB_CREATED_TIMEOUT_SECONDS))

  while (( SECONDS < deadline )); do
    if [[ -f "${log_path}" ]] && grep -q "✓ job created" "${log_path}"; then
      sed -n '1,/✓ job created/p' "${log_path}"
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

# Cx8: dense-ladder batch rule, 786,432 tokens / 96 sequences.
# Coarse factor-of-two LR sweep; use 4 GPUs while 275M capacity is ample.
launch_one 8 b768k 96 4 8 2e-4 lr2e-4
launch_one 8 b768k 96 4 8 4e-4 lr4e-4
launch_one 8 b768k 96 4 8 8e-4 lr8e-4
launch_one 8 b768k 96 4 8 1.6e-3 lr1.6e-3

# Cx16: 1,048,576 tokens / 128 sequences.
# Coarse factor-of-two LR sweep; use 4 GPUs for better turnaround.
launch_one 16 b1m 128 4 16 1e-4 lr1e-4
launch_one 16 b1m 128 4 16 2e-4 lr2e-4
launch_one 16 b1m 128 4 16 4e-4 lr4e-4
launch_one 16 b1m 128 4 16 8e-4 lr8e-4
