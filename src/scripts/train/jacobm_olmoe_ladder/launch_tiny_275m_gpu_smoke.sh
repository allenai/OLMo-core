#!/usr/bin/env bash
set -euo pipefail

SCRIPT="src/scripts/train/jacobm_olmoe_ladder/moe_a0_ladder.py"
RUN_PREFIX="olmoe3-tiny-275m"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3}"
LOG_DIR="${LOG_DIR:-/tmp/olmoe3-tiny-275m-gpu-smoke-launch-logs}"
JOB_CREATED_TIMEOUT_SECONDS="${JOB_CREATED_TIMEOUT_SECONDS:-240}"
NUM_NODES=1
NUM_GPUS=2
MICRO_BSZ=16

mkdir -p "${LOG_DIR}"

COMMON_BEAKER_ARGS=(
  --cluster ai2/titan
  --nodes "${NUM_NODES}"
  --gpus "${NUM_GPUS}"
  --weka oe-training-default
  --beaker-image tianhuat/olmo-core-torch211-2404-cu128
  --workspace ai2/OLMo-3-moe-experiments
  --budget ai2/oe-other
  --priority urgent
  --env OLMO_SYMM_VDEV2D_AUTO_BUILD=1
  --env-secret AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY WANDB_API_KEY=jacobm_WANDB_API_KEY
)

launch_one() {
  local lr="$1"
  local lr_tag="$2"
  local name="${RUN_PREFIX}-cx1-b256k-gpu${NUM_GPUS}-ep1mb${MICRO_BSZ}-${lr_tag}"
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
    --chinchilla-multiple=1
    --global-batch-size-seq=32
    --num-nodes="${NUM_NODES}"
    --gpus-per-node="${NUM_GPUS}"
    --micro-batch-size="${MICRO_BSZ}"
    --ep-dim=1
    --tag="${lr_tag}-cx1-b256k-gpu${NUM_GPUS}-ep1mb${MICRO_BSZ}"
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

launch_one 3e-3 lr3e-3
