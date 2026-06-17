#!/usr/bin/env bash
set -euo pipefail

SCRIPT="src/scripts/train/jacobm_olmoe_ladder/moe_a0_ladder.py"
RUN_PREFIX="${RUN_PREFIX:-sp}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/total_sparsity}"
LOG_DIR="${LOG_DIR:-/tmp/olmoe3-total-sparsity-480m-promoted-launch-logs}"
JOB_CREATED_TIMEOUT_SECONDS="${JOB_CREATED_TIMEOUT_SECONDS:-240}"
NUM_NODES="${NUM_NODES:-1}"
EP_DIM=1
CLUSTER="${CLUSTER:-ai2/titan}"
BEAKER_IMAGE="${BEAKER_IMAGE:-tianhuat/olmo-core-torch211-2404-cu128}"
WORKSPACE="${WORKSPACE:-ai2/OLMo-3-moe-experiments}"
BUDGET="${BUDGET:-ai2/oe-other}"
PRIORITY="${PRIORITY:-urgent}"
SWEEP_SUFFIX="${SWEEP_SUFFIX:-r1}"
EPHEMERAL_SAVE_INTERVAL="${EPHEMERAL_SAVE_INTERVAL:-500}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2000}"
SPARSITY_VARIANTS="${SPARSITY_VARIANTS:-high_total_96e_top4 huge_total_192e_top4}"
CX_LIST="${CX_LIST:-1 2 4 8}"

mkdir -p "${LOG_DIR}"

common_beaker_args=(
  --cluster "${CLUSTER}"
  --nodes "${NUM_NODES}"
  --weka oe-training-default
  --beaker-image "${BEAKER_IMAGE}"
  --workspace "${WORKSPACE}"
  --budget "${BUDGET}"
  --priority "${PRIORITY}"
  --env OLMO_SYMM_VDEV2D_AUTO_BUILD=1
  --env-secret AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY WANDB_API_KEY=jacobm_WANDB_API_KEY
)

tag_for() {
  case "$1" in
    high|sp96e4k|high_total_96e_top4) echo sp96e4k ;;
    huge|sp192e4k|huge_total_192e_top4) echo sp192e4k ;;
    *) echo "Unknown total sparsity selector: $1" >&2; return 1 ;;
  esac
}

canonical_variant() {
  case "$1" in
    high|sp96e4k|high_total_96e_top4) echo high_total_96e_top4 ;;
    huge|sp192e4k|huge_total_192e_top4) echo huge_total_192e_top4 ;;
    *) echo "Unknown total sparsity selector: $1" >&2; return 1 ;;
  esac
}

launch_one() {
  local total_sparsity="$1"
  local sp_tag="$2"
  local cx="$3"
  local batch_tag="$4"
  local global_batch_size_seq="$5"
  local gpus="$6"
  local micro_bsz="$7"
  local lr="$8"
  local lr_tag="$9"
  local denom=$((NUM_NODES * gpus * micro_bsz))
  if (( global_batch_size_seq % denom != 0 )); then
    echo "Invalid batch settings for ${sp_tag} Cx${cx}: global_batch_size_seq=${global_batch_size_seq} is not divisible by nodes*gpus*micro_bsz=${denom}" >&2
    exit 1
  fi

  local name="${RUN_PREFIX}-480m-cx${cx}-${sp_tag}-${lr_tag}-${SWEEP_SUFFIX}"
  local log_path="${LOG_DIR}/${name}.log"
  local systems_tag="${batch_tag}-gpu${gpus}-ep${EP_DIM}mb${micro_bsz}"

  local cmd=(
    uv run --extra dev --extra beaker python -m olmo_core.launch.beaker
    --allow-dirty
    --name="${name}"
    --gpus "${gpus}"
    "${common_beaker_args[@]}"
    --
    python "${SCRIPT}"
    --model-size=480m
    --total-sparsity="${total_sparsity}"
    --compile
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
    --tag="${sp_tag}-480m-cx${cx}-${lr_tag}-${SWEEP_SUFFIX}"
    --wandb-tag=exp_total_sparsity
    --wandb-tag="${sp_tag}"
    --wandb-tag=480m
    --wandb-tag="cx${cx}"
    --wandb-tag="${batch_tag}"
    --wandb-tag="${lr_tag}"
    --wandb-tag="${systems_tag}"
    --wandb-tag=compile-on
    --wandb-tag=lr-shifted-from-275m
    --wandb-tag=promoted-single-point
    --wandb-tag=titan
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

should_launch_cx() {
  local target="$1"
  for cx in ${CX_LIST}; do
    if [[ "${cx}" == "${target}" ]]; then
      return 0
    fi
  done
  return 1
}

launch_if_enabled() {
  local cx="$3"
  if should_launch_cx "${cx}"; then
    launch_one "$@"
  fi
}

launch_variant() {
  local total_sparsity="$1"
  local sp_tag="$2"

  case "${sp_tag}" in
    sp96e4k)
      launch_if_enabled "${total_sparsity}" "${sp_tag}" 1 b256k 32 4 8 1e-3 lr1e-3
      launch_if_enabled "${total_sparsity}" "${sp_tag}" 2 b384k 48 4 4 8e-4 lr8e-4
      launch_if_enabled "${total_sparsity}" "${sp_tag}" 4 b512k 64 4 8 7e-4 lr7e-4
      launch_if_enabled "${total_sparsity}" "${sp_tag}" 8 b768k 96 8 4 7e-4 lr7e-4
      ;;
    sp192e4k)
      launch_if_enabled "${total_sparsity}" "${sp_tag}" 1 b256k 32 4 8 8e-4 lr8e-4
      launch_if_enabled "${total_sparsity}" "${sp_tag}" 2 b384k 48 4 4 6e-4 lr6e-4
      launch_if_enabled "${total_sparsity}" "${sp_tag}" 4 b512k 64 4 4 6e-4 lr6e-4
      launch_if_enabled "${total_sparsity}" "${sp_tag}" 8 b768k 96 8 4 6e-4 lr6e-4
      ;;
    *)
      echo "Unsupported total sparsity tag: ${sp_tag}" >&2
      exit 1
      ;;
  esac
}

for variant in ${SPARSITY_VARIANTS}; do
  launch_variant "$(canonical_variant "${variant}")" "$(tag_for "${variant}")"
done
