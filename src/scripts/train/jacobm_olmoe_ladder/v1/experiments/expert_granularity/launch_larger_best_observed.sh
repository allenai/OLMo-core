#!/usr/bin/env bash
set -euo pipefail

SCRIPT="src/scripts/train/jacobm_olmoe_ladder/moe_a0_ladder.py"
RUN_PREFIX="${RUN_PREFIX:-eg}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/expert_granularity}"
LOG_DIR="${LOG_DIR:-/tmp/olmoe3-expert-granularity-larger-best-observed-launch-logs}"
JOB_CREATED_TIMEOUT_SECONDS="${JOB_CREATED_TIMEOUT_SECONDS:-240}"
NUM_NODES=1
EP_DIM=1
SWEEP_SUFFIX="${SWEEP_SUFFIX:-r1}"
EPHEMERAL_SAVE_INTERVAL="${EPHEMERAL_SAVE_INTERVAL:-500}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2000}"

mkdir -p "${LOG_DIR}"

common_beaker_args=(
  --cluster ai2/titan
  --nodes "${NUM_NODES}"
  --weka oe-training-default
  --beaker-image tianhuat/olmo-core-torch211-2404-cu128
  --workspace ai2/OLMo-3-moe-experiments
  --budget ai2/oe-other
  --priority urgent
  --env OLMO_SYMM_VDEV2D_AUTO_BUILD=1
  --env-secret AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY WANDB_API_KEY=jacobm_WANDB_API_KEY
)

launch_one() {
  local model_size="$1"
  local expert_geometry="$2"
  local eg_tag="$3"
  local cx="$4"
  local batch_tag="$5"
  local global_batch_size_seq="$6"
  local gpus="$7"
  local micro_bsz="$8"
  local lr="$9"
  local lr_tag="${10}"
  local name="${RUN_PREFIX}-${model_size}-cx${cx}-${eg_tag}-${lr_tag}-${SWEEP_SUFFIX}"
  local log_path="${LOG_DIR}/${name}.log"

  local cmd=(
    uv run --extra dev --extra beaker python -m olmo_core.launch.beaker
    --allow-dirty
    --name="${name}"
    --gpus "${gpus}"
    "${common_beaker_args[@]}"
    --
    python "${SCRIPT}"
    --model-size="${model_size}"
    --expert-geometry="${expert_geometry}"
    --save-folder="${CHECKPOINT_ROOT}/${name}"
    --name="${name}"
    --data-root=s3://ai2-llm
    --lr="${lr}"
    --chinchilla-multiple="${cx}"
    --global-batch-size-seq="${global_batch_size_seq}"
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
    --tag="${eg_tag}-${model_size}-cx${cx}-${lr_tag}-${SWEEP_SUFFIX}"
    --wandb-tag=exp_expert_granularity
    --wandb-tag="${eg_tag}"
    --wandb-tag="${model_size}"
    --wandb-tag="cx${cx}"
    --wandb-tag="${batch_tag}"
    --wandb-tag="${lr_tag}"
    --wandb-tag=baseline-best-observed
    --wandb-tag=promoted-single-point
    --wandb-tag="nodes${NUM_NODES}"
    --wandb-tag="gpu${gpus}"
    --wandb-tag="ep${EP_DIM}"
    --wandb-tag="mb${micro_bsz}"
    --wandb-tag=compute-efficient
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

for expert_spec in coarse_24e_top2:eg24e2k fine_96e_top8:eg96e8k; do
  expert_geometry="${expert_spec%%:*}"
  eg_tag="${expert_spec##*:}"

  launch_one 810m "${expert_geometry}" "${eg_tag}" 2 b384k 48 8 6 5.6e-4 lr5.6e-4
  launch_one 810m "${expert_geometry}" "${eg_tag}" 8 b768k 96 8 6 4e-4 lr4e-4
  launch_one 1p2b "${expert_geometry}" "${eg_tag}" 1 b256k 32 8 4 4e-4 lr4e-4
done
