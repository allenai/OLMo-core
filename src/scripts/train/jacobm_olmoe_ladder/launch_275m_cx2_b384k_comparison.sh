#!/usr/bin/env bash
set -euo pipefail

# Rerun 275M Cx2 with a smoother intermediate optimizer batch:
# 393,216 tokens = 48 sequences at sequence length 8192.
#
# This launches three comparable curves:
# - baseline A0;
# - expert granularity coarse, 24E/top2;
# - expert granularity fine, 96E/top8.

SCRIPT="src/scripts/train/jacobm_olmoe_ladder/tiny_275m.py"
BASE_CHECKPOINT_ROOT="${BASE_CHECKPOINT_ROOT:-/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3}"
EG_CHECKPOINT_ROOT="${EG_CHECKPOINT_ROOT:-/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/expert_granularity}"
LOG_DIR="${LOG_DIR:-/tmp/olmoe3-275m-cx2-b384k-comparison-launch-logs}"
JOB_CREATED_TIMEOUT_SECONDS="${JOB_CREATED_TIMEOUT_SECONDS:-240}"
NUM_NODES=1
GPUS=2
EP_DIM=1
MICRO_BSZ=8
GLOBAL_BATCH_SIZE_SEQ=48
CHINCHILLA_MULTIPLE=2
SWEEP_SUFFIX="${SWEEP_SUFFIX:-r1}"
EPHEMERAL_SAVE_INTERVAL="${EPHEMERAL_SAVE_INTERVAL:-500}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2000}"
LR_SPECS="${LR_SPECS:-9e-4:lr9e-4 1.8e-3:lr1.8e-3 3.6e-3:lr3.6e-3}"

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
  local family="$1"
  local lr="$2"
  local lr_tag="$3"

  local name=""
  local save_root=""
  local extra_model_args=()
  local wandb_tags=()

  case "${family}" in
    baseline)
      name="olmoe3-tiny-275m-cx2-b384k-gpu${GPUS}-ep${EP_DIM}mb${MICRO_BSZ}-${lr_tag}-${SWEEP_SUFFIX}"
      save_root="${BASE_CHECKPOINT_ROOT}"
      wandb_tags=(baseline b384k-cx2)
      ;;
    eg24e2k)
      name="eg-275m-cx2-b384k-eg24e2k-${lr_tag}-${SWEEP_SUFFIX}"
      save_root="${EG_CHECKPOINT_ROOT}"
      extra_model_args=(--expert-geometry=coarse_24e_top2)
      wandb_tags=(exp_expert_granularity eg24e2k b384k-cx2 baseline-comparison)
      ;;
    eg96e8k)
      name="eg-275m-cx2-b384k-eg96e8k-${lr_tag}-${SWEEP_SUFFIX}"
      save_root="${EG_CHECKPOINT_ROOT}"
      extra_model_args=(--expert-geometry=fine_96e_top8)
      wandb_tags=(exp_expert_granularity eg96e8k b384k-cx2 baseline-comparison)
      ;;
    *)
      echo "Unknown family: ${family}" >&2
      exit 1
      ;;
  esac

  local log_path="${LOG_DIR}/${name}.log"
  local systems_tag="b384k-gpu${GPUS}-ep${EP_DIM}mb${MICRO_BSZ}"

  local cmd=(
    uv run --extra dev --extra beaker python -m olmo_core.launch.beaker
    --allow-dirty
    --name="${name}"
    "${common_beaker_args[@]}"
    --
    python "${SCRIPT}"
    --model-size=275m
    "${extra_model_args[@]}"
    --save-folder="${save_root}/${name}"
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
    --tag="${family}-cx2-${lr_tag}-${SWEEP_SUFFIX}"
    --wandb-tag=275m
    --wandb-tag=cx2
    --wandb-tag="${lr_tag}"
    --wandb-tag="${systems_tag}"
  )

  for tag in "${wandb_tags[@]}"; do
    cmd+=(--wandb-tag="${tag}")
  done

  echo "Launching ${name}..."
  printf 'Command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'

  "${cmd[@]}" >"${log_path}" 2>&1 &
  local pid=$!
  wait_for_job_created "${name}" "${log_path}" "${pid}"
}

for family in baseline eg24e2k eg96e8k; do
  for lr_spec in ${LR_SPECS}; do
    launch_one "${family}" "${lr_spec%%:*}" "${lr_spec##*:}"
  done
done
