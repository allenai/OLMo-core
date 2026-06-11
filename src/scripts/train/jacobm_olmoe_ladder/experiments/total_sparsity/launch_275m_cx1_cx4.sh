#!/usr/bin/env bash
set -euo pipefail

SCRIPT="src/scripts/train/jacobm_olmoe_ladder/tiny_275m.py"
RUN_PREFIX="sp-275m"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/total_sparsity}"
LOG_DIR="${LOG_DIR:-/tmp/olmoe3-total-sparsity-275m-cx1-cx4-launch-logs}"
JOB_CREATED_TIMEOUT_SECONDS="${JOB_CREATED_TIMEOUT_SECONDS:-240}"
NUM_NODES=1
GPUS="${GPUS:-4}"
EP_DIM=1
MICRO_BSZ="${MICRO_BSZ:-4}"
SWEEP_SUFFIX="${SWEEP_SUFFIX:-r1}"
EPHEMERAL_SAVE_INTERVAL="${EPHEMERAL_SAVE_INTERVAL:-500}"
EVAL_INTERVAL="${EVAL_INTERVAL:-2000}"
SPARSITY_VARIANTS="${SPARSITY_VARIANTS:-high huge}"
CX_LIST="${CX_LIST:-1 4}"
CX1_LR_SPECS="${CX1_LR_SPECS:-1e-3:lr1e-3 2e-3:lr2e-3 4e-3:lr4e-3}"
CX4_LR_SPECS="${CX4_LR_SPECS:-8e-4:lr8e-4 1.6e-3:lr1.6e-3 3.2e-3:lr3.2e-3}"

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
  local cx="$3"
  local batch_tag="$4"
  local global_batch_size_seq="$5"
  local lr="$6"
  local lr_tag="$7"
  local name="${RUN_PREFIX}-cx${cx}-${sp_tag}-${lr_tag}-${SWEEP_SUFFIX}"
  local log_path="${LOG_DIR}/${name}.log"
  local systems_tag="${batch_tag}-gpu${GPUS}-ep${EP_DIM}mb${MICRO_BSZ}"

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
    --lr="${lr}"
    --chinchilla-multiple="${cx}"
    --global-batch-size-seq="${global_batch_size_seq}"
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
    --tag="${sp_tag}-cx${cx}-${lr_tag}-${SWEEP_SUFFIX}"
    --wandb-tag=exp_total_sparsity
    --wandb-tag="${sp_tag}"
    --wandb-tag=275m
    --wandb-tag="cx${cx}"
    --wandb-tag="${lr_tag}"
    --wandb-tag="${systems_tag}"
    --wandb-tag=baseline-transferred
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

variant_specs=()
for variant in ${SPARSITY_VARIANTS}; do
  case "${variant}" in
    high|sp96e4k|high_total_96e_top4)
      variant_specs+=("high_total_96e_top4:sp96e4k")
      ;;
    huge|sp192e4k|huge_total_192e_top4)
      variant_specs+=("huge_total_192e_top4:sp192e4k")
      ;;
    *)
      echo "Unknown total sparsity selector: ${variant}" >&2
      exit 1
      ;;
  esac
done

for spec in "${variant_specs[@]}"; do
  total_sparsity="${spec%%:*}"
  sp_tag="${spec##*:}"
  for cx in ${CX_LIST}; do
    case "${cx}" in
      1)
        for lr_spec in ${CX1_LR_SPECS}; do
          launch_one "${total_sparsity}" "${sp_tag}" 1 b256k 32 "${lr_spec%%:*}" "${lr_spec##*:}"
        done
        ;;
      4)
        for lr_spec in ${CX4_LR_SPECS}; do
          launch_one "${total_sparsity}" "${sp_tag}" 4 b512k 64 "${lr_spec%%:*}" "${lr_spec##*:}"
        done
        ;;
      *)
        echo "Unsupported Cx for this launcher: ${cx}" >&2
        exit 1
        ;;
    esac
  done
done
