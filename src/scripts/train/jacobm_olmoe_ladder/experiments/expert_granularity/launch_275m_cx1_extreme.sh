#!/usr/bin/env bash
set -euo pipefail

SCRIPT="src/scripts/train/jacobm_olmoe_ladder/tiny_275m.py"
RUN_PREFIX="eg-275m-cx1"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/expert_granularity}"
LOG_DIR="${LOG_DIR:-/tmp/olmoe3-expert-granularity-275m-cx1-extreme-launch-logs}"
JOB_CREATED_TIMEOUT_SECONDS="${JOB_CREATED_TIMEOUT_SECONDS:-240}"
NUM_NODES=1
GPUS=1
EP_DIM=1
EXTREME_MICRO_BSZ="${EXTREME_MICRO_BSZ:-4}"
ULTRA_MICRO_BSZ="${ULTRA_MICRO_BSZ:-2}"
GLOBAL_BATCH_SIZE_SEQ=32
CHINCHILLA_MULTIPLE=1
SWEEP_SUFFIX="${SWEEP_SUFFIX:-r1}"
EPHEMERAL_SAVE_INTERVAL="${EPHEMERAL_SAVE_INTERVAL:-500}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2000}"

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
  local lr="$3"
  local lr_tag="$4"
  local micro_bsz="$5"
  local name="${RUN_PREFIX}-${eg_tag}-${lr_tag}-${SWEEP_SUFFIX}"
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
    --lr="${lr}"
    --chinchilla-multiple="${CHINCHILLA_MULTIPLE}"
    --global-batch-size-seq="${GLOBAL_BATCH_SIZE_SEQ}"
    --num-nodes="${NUM_NODES}"
    --gpus-per-node="${GPUS}"
    --micro-batch-size="${micro_bsz}"
    --ep-dim="${EP_DIM}"
    --ladder-evals
    --eval-task-set=fast
    --eval-interval="${EVAL_INTERVAL}"
    --save-interval=999999999
    --ephemeral-save-interval="${EPHEMERAL_SAVE_INTERVAL}"
    --no-pre-train-checkpoint
    --tag="${eg_tag}-cx1-${lr_tag}-${SWEEP_SUFFIX}"
    --wandb-tag=exp_expert_granularity
    --wandb-tag="${eg_tag}"
    --wandb-tag=275m
    --wandb-tag=cx1
    --wandb-tag="${lr_tag}"
    --wandb-tag="${systems_tag}"
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

for spec in extreme_192e_top16:eg192e16k ultra_384e_top32:eg384e32k; do
  expert_geometry="${spec%%:*}"
  eg_tag="${spec##*:}"
  micro_bsz="${EXTREME_MICRO_BSZ}"
  if [[ "${eg_tag}" == "eg384e32k" ]]; then
    micro_bsz="${ULTRA_MICRO_BSZ}"
  fi
  launch_one "${expert_geometry}" "${eg_tag}" 1e-3 lr1e-3 "${micro_bsz}"
  launch_one "${expert_geometry}" "${eg_tag}" 2e-3 lr2e-3 "${micro_bsz}"
  launch_one "${expert_geometry}" "${eg_tag}" 4e-3 lr4e-3 "${micro_bsz}"
done
