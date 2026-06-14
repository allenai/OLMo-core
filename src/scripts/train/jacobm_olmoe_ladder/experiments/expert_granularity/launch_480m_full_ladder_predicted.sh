#!/usr/bin/env bash
set -euo pipefail

# 480M expert-granularity full Cx1/Cx2/Cx4/Cx8 ladder at predicted LRs.
# Names are semantic/resume-stable: systems settings are W&B tags/config only.

SCRIPT="src/scripts/train/jacobm_olmoe_ladder/moe_a0_ladder.py"
RUN_PREFIX="eg-480m"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/expert_granularity}"
LOG_DIR="${LOG_DIR:-/tmp/olmoe3-expert-granularity-480m-full-ladder-launch-logs}"
JOB_CREATED_TIMEOUT_SECONDS="${JOB_CREATED_TIMEOUT_SECONDS:-240}"
NUM_NODES="${NUM_NODES:-1}"
EP_DIM="${EP_DIM:-1}"
SWEEP_SUFFIX="${SWEEP_SUFFIX:-r1}"
EPHEMERAL_SAVE_INTERVAL="${EPHEMERAL_SAVE_INTERVAL:-500}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2000}"
EXPERT_VARIANTS="${EXPERT_VARIANTS:-coarse fine}"
CX_LIST="${CX_LIST:-1 2 4 8}"

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
  local expert_geometry="$1"
  local eg_tag="$2"
  local cx="$3"
  local batch_tag="$4"
  local global_batch_size_seq="$5"
  local gpus="$6"
  local micro_bsz="$7"
  local lr="$8"
  local lr_tag="$9"
  local name="${RUN_PREFIX}-cx${cx}-${eg_tag}-${lr_tag}-${SWEEP_SUFFIX}"
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
    --name="${name}"
    "${common_beaker_args[@]}"
    --
    python "${SCRIPT}"
    --model-size=480m
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
    --tag="${eg_tag}-480m-cx${cx}-${lr_tag}-${SWEEP_SUFFIX}"
    --wandb-tag=exp_expert_granularity
    --wandb-tag="${eg_tag}"
    --wandb-tag=480m
    --wandb-tag="cx${cx}"
    --wandb-tag="${batch_tag}"
    --wandb-tag="${lr_tag}"
    --wandb-tag=predicted-lr
    --wandb-tag=full-ladder
    --wandb-tag=nodes${NUM_NODES}
    --wandb-tag=gpu${gpus}
    --wandb-tag=ep${EP_DIM}
    --wandb-tag=mb${micro_bsz}
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

variant_specs=()
for variant in ${EXPERT_VARIANTS}; do
  case "${variant}" in
    coarse|eg24e2k|coarse_24e_top2)
      variant_specs+=("coarse_24e_top2:eg24e2k")
      ;;
    fine|eg96e8k|fine_96e_top8)
      variant_specs+=("fine_96e_top8:eg96e8k")
      ;;
    *)
      echo "Unknown expert-geometry selector: ${variant}" >&2
      exit 1
      ;;
  esac
done

for spec in "${variant_specs[@]}"; do
  expert_geometry="${spec%%:*}"
  eg_tag="${spec##*:}"
  for cx in ${CX_LIST}; do
    case "${eg_tag}:${cx}" in
      eg24e2k:1) launch_one "${expert_geometry}" "${eg_tag}" 1 b256k 32 4 8 9e-4 lr9e-4 ;;
      eg24e2k:2) launch_one "${expert_geometry}" "${eg_tag}" 2 b384k 48 4 4 1e-3 lr1e-3 ;;
      eg24e2k:4) launch_one "${expert_geometry}" "${eg_tag}" 4 b512k 64 4 8 8e-4 lr8e-4 ;;
      eg24e2k:8) launch_one "${expert_geometry}" "${eg_tag}" 8 b768k 96 8 4 8e-4 lr8e-4 ;;
      eg96e8k:1) launch_one "${expert_geometry}" "${eg_tag}" 1 b256k 32 4 8 1e-3 lr1e-3 ;;
      eg96e8k:2) launch_one "${expert_geometry}" "${eg_tag}" 2 b384k 48 4 4 1e-3 lr1e-3 ;;
      eg96e8k:4) launch_one "${expert_geometry}" "${eg_tag}" 4 b512k 64 4 8 8e-4 lr8e-4 ;;
      eg96e8k:8) launch_one "${expert_geometry}" "${eg_tag}" 8 b768k 96 8 4 8e-4 lr8e-4 ;;
      *)
        echo "Unsupported variant/Cx combination: ${eg_tag} Cx${cx}" >&2
        exit 1
        ;;
    esac
  done
done
