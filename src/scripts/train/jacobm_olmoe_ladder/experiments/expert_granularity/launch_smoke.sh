#!/usr/bin/env bash
set -euo pipefail

SCRIPT="src/scripts/train/jacobm_olmoe_ladder/tiny_275m.py"
RUN_PREFIX="eg-smoke"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/expert_granularity}"
LOG_DIR="${LOG_DIR:-/tmp/olmoe3-expert-granularity-smoke-launch-logs}"
JOB_CREATED_TIMEOUT_SECONDS="${JOB_CREATED_TIMEOUT_SECONDS:-240}"
NUM_NODES=1
GPUS=1
EP_DIM=1
COARSE_MICRO_BSZ="${COARSE_MICRO_BSZ:-16}"
FINE_MICRO_BSZ="${FINE_MICRO_BSZ:-8}"
EXTREME_MICRO_BSZ="${EXTREME_MICRO_BSZ:-4}"
ULTRA_MICRO_BSZ="${ULTRA_MICRO_BSZ:-2}"
GLOBAL_BATCH_SIZE_SEQ=32
CHINCHILLA_MULTIPLE="${CHINCHILLA_MULTIPLE:-0.02}"
LR="${LR:-2e-3}"
LR_TAG="${LR_TAG:-lr2e-3}"
SMOKE_SUFFIX="${SMOKE_SUFFIX:-r1}"
EPHEMERAL_SAVE_INTERVAL="${EPHEMERAL_SAVE_INTERVAL:-500}"
EXPERT_GEOMETRIES="${EXPERT_GEOMETRIES:-coarse fine}"

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
  local expert_geometry="$1"
  local eg_tag="$2"
  local micro_bsz="$3"
  local name="${RUN_PREFIX}-${eg_tag}-${LR_TAG}-${SMOKE_SUFFIX}"
  local log_path="${LOG_DIR}/${name}.log"
  local systems_tag="b256k-gpu${GPUS}-ep${EP_DIM}mb${micro_bsz}"

  local cmd=(
    uv run --extra dev --extra beaker python -m olmo_core.launch.beaker
    --allow-dirty
    --name="${name}"
    "${common_beaker_args[@]}"
    --
    python "${SCRIPT}"
    --model-size=275m
    --expert-geometry="${expert_geometry}"
    --save-folder="${CHECKPOINT_ROOT}/${name}"
    --name="${name}"
    --data-root=s3://ai2-llm
    --lr="${LR}"
    --chinchilla-multiple="${CHINCHILLA_MULTIPLE}"
    --global-batch-size-seq="${GLOBAL_BATCH_SIZE_SEQ}"
    --num-nodes="${NUM_NODES}"
    --gpus-per-node="${GPUS}"
    --micro-batch-size="${micro_bsz}"
    --ep-dim="${EP_DIM}"
    --save-interval=999999999
    --ephemeral-save-interval="${EPHEMERAL_SAVE_INTERVAL}"
    --no-pre-train-checkpoint
    --tag="${eg_tag}-smk-${LR_TAG}-${SMOKE_SUFFIX}"
    --wandb-tag=exp_expert_granularity
    --wandb-tag="${eg_tag}"
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

for variant in ${EXPERT_GEOMETRIES}; do
  case "${variant}" in
    coarse|eg24e2k|coarse_24e_top2)
      launch_one coarse_24e_top2 eg24e2k "${COARSE_MICRO_BSZ}"
      ;;
    fine|eg96e8k|fine_96e_top8)
      launch_one fine_96e_top8 eg96e8k "${FINE_MICRO_BSZ}"
      ;;
    extreme|eg192e16k|extreme_192e_top16)
      launch_one extreme_192e_top16 eg192e16k "${EXTREME_MICRO_BSZ}"
      ;;
    ultra|eg384e32k|ultra_384e_top32)
      launch_one ultra_384e_top32 eg384e32k "${ULTRA_MICRO_BSZ}"
      ;;
    *)
      echo "Unknown expert geometry selector: ${variant}" >&2
      exit 1
      ;;
  esac
done
