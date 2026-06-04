#!/usr/bin/env bash
set -euo pipefail

# Dry-run by default. Run with DRY_RUN=0 to delete:
#   DRY_RUN=0 bash src/scripts/train/jacobm_olmoe_ladder/rm_discussed_checkpoints.sh
DRY_RUN="${DRY_RUN:-1}"
KEEP_STEP_CHECKPOINTS="${KEEP_STEP_CHECKPOINTS:-2}"

paths=(
  # Failed/incomplete Cx8/Cx16 relaunches from the full /weka incident.
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx8-b768k-gpu4-ep1mb8-lr2e-4
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx8-b768k-gpu4-ep1mb8-lr4e-4
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx8-b768k-gpu4-ep1mb8-lr8e-4
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx8-b768k-gpu4-ep1mb8-lr1.6e-3
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx16-b1m-gpu4-ep1mb16-lr1e-4
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx16-b1m-gpu4-ep1mb16-lr2e-4
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx16-b1m-gpu4-ep1mb16-lr4e-4
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx16-b1m-gpu4-ep1mb16-lr8e-4

  # Stopped/replaced old granular Cx8/Cx16 runs.
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx8-b768k-gpu2-ep1mb16-lr3e-4
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx8-b768k-gpu2-ep1mb16-lr5e-4
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx8-b768k-gpu2-ep1mb16-lr7e-4
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx8-b768k-gpu2-ep1mb16-lr1e-3
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx16-b1m-gpu2-ep1mb16-lr2e-4
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx16-b1m-gpu2-ep1mb16-lr3e-4
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx16-b1m-gpu2-ep1mb16-lr5e-4
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx16-b1m-gpu2-ep1mb16-lr7e-4

  # Smoke checkpoints.
  /weka/oe-training-default/ai2-llm/checkpoints/jacob/olmoe3-tiny-275m-4xchinchilla-smoketest
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx8-smoke-b768k-gpu2-ep1mb24-lr5e-4
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx8-smoke-b768k-gpu2-ep1mb16-lr5e-4
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx16-smoke-b1m-gpu2-ep1mb32-lr5e-4
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx16-smoke-b1m-gpu2-ep1mb16-lr5e-4
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-moe-a0-810m-smoke-b256k-gpu2-ep1mb16-lr5e-4-r2
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-moe-a0-810m-smoke-b256k-gpu2-ep1mb8-lr5e-4-r2
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-moe-a0-810m-smoke-b256k-gpu4-ep1mb8-lr5e-4-r3
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-moe-a0-810m-smoke-b256k-gpu4-ep1mb4-lr5e-4-r3
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-moe-a0-1p2b-smoke-b256k-gpu2-ep1mb8-lr3e-4-r2
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-moe-a0-1p2b-smoke-b256k-gpu2-ep1mb4-lr3e-4-r2
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-moe-a0-1p2b-smoke-b256k-gpu4-ep1mb4-lr3e-4-r3
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-moe-a0-1p2b-smoke-b256k-gpu4-ep2mb4-lr3e-4-r3

  # Superseded/non-middle Cx1 batch probes and far-tail LRs.
  /weka/oe-training-default/ai2-llm/checkpoints/jacob/olmoe3-tiny-275m-cx1-b128k-lr5e-4
  /weka/oe-training-default/ai2-llm/checkpoints/jacob/olmoe3-tiny-275m-cx1-b128k-lr8e-4
  /weka/oe-training-default/ai2-llm/checkpoints/jacob/olmoe3-tiny-275m-cx1-b512k-lr5e-4
  /weka/oe-training-default/ai2-llm/checkpoints/jacob/olmoe3-tiny-275m-cx1-b512k-lr8e-4
  /weka/oe-training-default/ai2-llm/checkpoints/jacob/olmoe3-tiny-275m-cx1-lr1e-4
  /weka/oe-training-default/ai2-llm/checkpoints/jacob/olmoe3-tiny-275m-cx1-lr3e-4
  /weka/oe-training-default/ai2-llm/checkpoints/jacob/olmoe3-tiny-275m-cx1-lr8e-4
  /weka/oe-training-default/ai2-llm/checkpoints/jacob/olmoe3-tiny-275m-cx1-lr1.2e-3
  /weka/oe-training-default/ai2-llm/checkpoints/jacob/olmoe3-tiny-275m-cx1-b256k-lr3e-4
  /weka/oe-training-default/ai2-llm/checkpoints/jacob/olmoe3-tiny-275m-cx1-b256k-lr4e-4
  /weka/oe-training-default/ai2-llm/checkpoints/jacob/olmoe3-tiny-275m-cx1-b256k-lr5e-4
  /weka/oe-training-default/ai2-llm/checkpoints/jacob/olmoe3-tiny-275m-cx1-b256k-lr6e-4
  /weka/oe-training-default/ai2-llm/checkpoints/jacob/olmoe3-tiny-275m-cx1-b256k-lr7e-4
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx1-b256k-gpu2-ep1mb16-lr3e-3
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx1-b256k-gpu2-ep1mb16-lr5e-3
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx1-b256k-gpu2-ep1mb16-lr8e-3

  # Superseded/non-middle Cx2 and Cx4 checkpoints.
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx2-b256k-ep1mb4-lr1e-3
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx2-b256k-lr1e-3
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx2-b256k-gpu2-ep1mb16-lr3.5e-3
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx4-b512k-ep1mb8-lr1e-3
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx4-b512k-ep1mb8-lr1.5e-3
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx4-b512k-ep1mb8-lr2.5e-3
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx4-b512k-lr1e-3
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx4-b512k-lr1.5e-3
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx4-b512k-lr2.5e-3
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx4-b512k-gpu4-ep1mb16-lr5e-4
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx4-b512k-gpu4-ep1mb16-lr5e-3

  # More aggressive storage-pressure tier: remove edge/redundant checkpoints
  # while keeping best/near-best anchors for each completed curve.
  /weka/oe-training-default/ai2-llm/checkpoints/jacob/olmoe3-tiny-275m-cx1-b256k-lr8e-4
  /weka/oe-training-default/ai2-llm/checkpoints/jacob/olmoe3-tiny-275m-cx1-b256k-lr1e-3
  /weka/oe-training-default/ai2-llm/checkpoints/jacob/olmoe3-tiny-275m-cx1-b256k-lr1.2e-3
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx2-b256k-gpu2-ep1mb16-lr1.5e-3
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx2-b256k-gpu2-ep1mb16-lr2.5e-3
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx4-b512k-gpu4-ep1mb16-lr7e-4
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx4-b512k-gpu4-ep1mb16-lr3.5e-3
)

anchor_run_dirs=(
  # Keep these run directories, but prune old intermediate step checkpoints
  # from them. This approximates the new ephemeral checkpoint behavior while
  # preserving the latest two checkpoints for inspection/resume safety.
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx1-b256k-n2-lr1.5e-3
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx1-b256k-n2-lr2e-3
  /weka/oe-training-default/ai2-llm/checkpoints/jacob/olmoe3-tiny-275m-cx2-b256k-lr5e-4
  /weka/oe-training-default/ai2-llm/checkpoints/jacob/olmoe3-tiny-275m-cx2-b256k-lr7e-4
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx2-b256k-gpu2-ep1mb16-lr1e-3
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx4-b512k-gpu4-ep1mb16-lr1e-3
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx4-b512k-gpu4-ep1mb16-lr1.5e-3
  /weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmoe3-tiny-275m-cx4-b512k-gpu4-ep1mb16-lr2.5e-3
)

prune_step_checkpoints() {
  local run_dir="$1"
  local keep="$2"
  local step_dirs=()

  if [[ ! -d "${run_dir}" ]]; then
    echo "Missing anchor run dir, skipping step prune ${run_dir}"
    return
  fi

  mapfile -t step_dirs < <(
    find "${run_dir}" -maxdepth 1 -mindepth 1 -type d -regextype posix-extended -regex '.*/step[0-9]+' -printf '%p\n' \
      | sort -V
  )

  if (( ${#step_dirs[@]} <= keep )); then
    echo "Keeping all ${#step_dirs[@]} step checkpoints in ${run_dir}"
    return
  fi

  local delete_count=$(( ${#step_dirs[@]} - keep ))
  for (( i = 0; i < delete_count; i++ )); do
    local step_path="${step_dirs[$i]}"
    if [[ "${DRY_RUN}" == "0" ]]; then
      echo "Deleting old step checkpoint ${step_path}"
      rm -rf -- "${step_path}"
    else
      echo "Would delete old step checkpoint ${step_path}"
    fi
  done

  for (( i = delete_count; i < ${#step_dirs[@]}; i++ )); do
    echo "Keeping recent step checkpoint ${step_dirs[$i]}"
  done
}

echo "DRY_RUN=${DRY_RUN}"
echo "KEEP_STEP_CHECKPOINTS=${KEEP_STEP_CHECKPOINTS}"
for path in "${paths[@]}"; do
  if [[ -e "${path}" ]]; then
    if [[ "${DRY_RUN}" == "0" ]]; then
      echo "Deleting ${path}"
      rm -rf -- "${path}"
    else
      echo "Would delete ${path}"
    fi
  else
    echo "Missing, skipping ${path}"
  fi
done

for run_dir in "${anchor_run_dirs[@]}"; do
  prune_step_checkpoints "${run_dir}" "${KEEP_STEP_CHECKPOINTS}"
done
