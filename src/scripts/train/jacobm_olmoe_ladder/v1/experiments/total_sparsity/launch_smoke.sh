#!/usr/bin/env bash
set -euo pipefail

SCRIPT="src/scripts/train/jacobm_olmoe_ladder/moe_a0_ladder.py"
RUN_PREFIX="sp-smoke"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/total_sparsity}"
LOG_DIR="${LOG_DIR:-/tmp/olmoe3-total-sparsity-smoke-launch-logs}"
JOB_CREATED_TIMEOUT_SECONDS="${JOB_CREATED_TIMEOUT_SECONDS:-240}"
NUM_NODES=1
GPUS="${GPUS:-4}"
EP_DIM=1
MICRO_BSZ="${MICRO_BSZ:-4}"
GLOBAL_BATCH_SIZE_SEQ=32
CHINCHILLA_MULTIPLE="${CHINCHILLA_MULTIPLE:-0.02}"
LR="${LR:-2e-3}"
LR_TAG="${LR_TAG:-lr2e-3}"
SMOKE_SUFFIX="${SMOKE_SUFFIX:-r1}"
EPHEMERAL_SAVE_INTERVAL="${EPHEMERAL_SAVE_INTERVAL:-500}"
SPARSITY_VARIANTS="${SPARSITY_VARIANTS:-high huge}"

mkdir -p "${LOG_DIR}"

common_beaker_args=(
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

launch_one() {
  local total_sparsity="$1"
  local sp_tag="$2"
  local name="${RUN_PREFIX}-${sp_tag}-${LR_TAG}-${SMOKE_SUFFIX}"
  local log_path="${LOG_DIR}/${name}.log"
  local systems_tag="b256k-gpu${GPUS}-ep${EP_DIM}mb${MICRO_BSZ}"

  local cmd=(
    uv run --extra dev --extra beaker python -m olmo_core.launch.beaker
    --allow-dirty
    --name="${name}"
    "${common_beaker_args[@]}"
    --
    python "${SCRIPT}"
    --model-size=275m
    --total-sparsity="${total_sparsity}"
    --save-folder="${CHECKPOINT_ROOT}/${name}"
    --name="${name}"
    --data-root=s3://ai2-llm
    --lr="${LR}"
    --chinchilla-multiple="${CHINCHILLA_MULTIPLE}"
    --global-batch-size-seq="${GLOBAL_BATCH_SIZE_SEQ}"
    --num-nodes="${NUM_NODES}"
    --gpus-per-node="${GPUS}"
    --micro-batch-size="${MICRO_BSZ}"
    --ep-dim="${EP_DIM}"
    --save-interval=999999999
    --ephemeral-save-interval="${EPHEMERAL_SAVE_INTERVAL}"
    --no-pre-train-checkpoint
    --tag="${sp_tag}-smk-${LR_TAG}-${SMOKE_SUFFIX}"
    --wandb-tag=exp_total_sparsity
    --wandb-tag="${sp_tag}"
    --wandb-tag=275m
    --wandb-tag=cx1
    --wandb-tag="${LR_TAG}"
    --wandb-tag="${systems_tag}"
    --wandb-tag=smoke
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

for variant in ${SPARSITY_VARIANTS}; do
  case "${variant}" in
    high|sp96e4k|high_total_96e_top4)
      launch_one high_total_96e_top4 sp96e4k
      ;;
    huge|sp192e4k|huge_total_192e_top4)
      launch_one huge_total_192e_top4 sp192e4k
      ;;
    *)
      echo "Unknown total sparsity selector: ${variant}" >&2
      exit 1
      ;;
  esac
done
