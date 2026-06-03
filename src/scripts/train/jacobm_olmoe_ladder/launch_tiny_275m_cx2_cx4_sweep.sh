#!/usr/bin/env bash
set -euo pipefail

SCRIPT="src/scripts/train/jacobm_olmoe_ladder/tiny_275m.py"
RUN_PREFIX="olmoe3-tiny-275m"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3}"
LOG_DIR="${LOG_DIR:-/tmp/olmoe3-tiny-275m-cx2-cx4-launch-logs}"
JOB_CREATED_TIMEOUT_SECONDS="${JOB_CREATED_TIMEOUT_SECONDS:-240}"
NUM_NODES=1

mkdir -p "${LOG_DIR}"

COMMON_BEAKER_ARGS=(
  --cluster ai2/titan
  --nodes "${NUM_NODES}"
  --gpus 8
  --weka oe-training-default
  --beaker-image tianhuat/olmo-core-torch211-2404-cu128
  --workspace ai2/OLMo-3-moe-experiments
  --budget ai2/oe-other
  --priority urgent
  --env OLMO_SYMM_VDEV2D_AUTO_BUILD=1
  --env-secret AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY WANDB_API_KEY=jacobm_WANDB_API_KEY
)

launch_one() {
  local chinchilla="$1"
  local batch_tag="$2"
  local batch_seq="$3"
  local lr="$4"
  local lr_tag="$5"
  local name="${RUN_PREFIX}-cx${chinchilla}-${batch_tag}-${lr_tag}"
  local log_path="${LOG_DIR}/${name}.log"

  local cmd=(
    uv run --extra dev --extra beaker python -m olmo_core.launch.beaker
    --name="${name}" \
    "${COMMON_BEAKER_ARGS[@]}" \
    -- \
    python "${SCRIPT}"
    --save-folder="${CHECKPOINT_ROOT}/${name}"
    --name="${name}"
    --data-root=s3://ai2-llm
    --lr="${lr}"
    --chinchilla-multiple="${chinchilla}"
    --global-batch-size-seq="${batch_seq}"
    --num-nodes="${NUM_NODES}"
    --tag="${lr_tag}-cx${chinchilla}-${batch_tag}"
  )

  echo "Launching ${name}..."
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'

  "${cmd[@]}" >"${log_path}" 2>&1 &
  local pid=$!
  local deadline=$((SECONDS + JOB_CREATED_TIMEOUT_SECONDS))

  while (( SECONDS < deadline )); do
    if grep -q "✓ job created" "${log_path}"; then
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

# Cx2 high-side check after 7e-4 beat 5e-4.
launch_one 2 b256k 32 1e-3 lr1e-3

# Cx4 sweep at the dense-ladder Cx4 batch rule: 512k tokens / 64 sequences.
launch_one 4 b512k 64 1e-3 lr1e-3
launch_one 4 b512k 64 1.5e-3 lr1.5e-3
launch_one 4 b512k 64 2.5e-3 lr2.5e-3
launch_one 4 b512k 64 3.5e-3 lr3.5e-3
