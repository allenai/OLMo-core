#!/usr/bin/env bash
#
# Scratch launcher for the MoE-v2 benchmark scripts under src/scripts/benchmarks/.
# Uses the generic Beaker launcher (`python -m olmo_core.launch.beaker ... -- CMD`).
#
# Usage:
#   ./launch_benches.sh                 # launch all benches
#   ./aunch_benches.sh odc_bench swiglu_valid_prefix_bench   # only these
#   DRY_RUN=1 ./launch_benches.sh       # print launch configs, don't submit
#
# Configure via env vars (or edit the defaults below).
set -euo pipefail

# ----------------------------------------------------------------------------- config
BEAKER_WORKSPACE="${BEAKER_WORKSPACE:-ai2/OLMo-core}"
BEAKER_BUDGET="${BEAKER_BUDGET:-ai2/oe-other}"
BEAKER_CLUSTER="${BEAKER_CLUSTER:-ai2/jupiter-cirrascale-2}"
BEAKER_PRIORITY="${BEAKER_PRIORITY:-normal}"       # low | normal | high | urgent
BEAKER_WEKA="${BEAKER_WEKA:-oe-training-default}"
RUN_PREFIX="${RUN_PREFIX:-moe-bench}"
DRY_RUN="${DRY_RUN:-0}"

# Image for the plain (single-GPU) benches. Empty -> launcher default (stable).
BEAKER_IMAGE="${BEAKER_IMAGE:-akshitab/olmo-core-tch2100cu128-rma-2026-07-08}"

# Image for the rowwise / symmetric-memory comm benches. These JIT-build the
# `symm_mem_vdev2d` CUDA extension at runtime, so this MUST be an image with the
# CUDA devel toolchain + NVSHMEM (the release/stable image will not work). Empty
# -> those benches are skipped with a warning.
BEAKER_SYMM_IMAGE="${BEAKER_SYMM_IMAGE:-}"

# Optional Beaker env secrets, space-separated "NAME=SECRET_NAME" pairs.
# e.g. BEAKER_ENV_SECRETS="WANDB_API_KEY=my-wandb-secret"
read -r -a BEAKER_ENV_SECRETS <<< "${BEAKER_ENV_SECRETS:-}"

# NOTE: the FP8 benches (mxfp8_*, scaled_grouped_mm_q, routed_experts_fp8, rowwise_fp8_comm)
# exercise the MXFP8 scaled-grouped-mm path, which needs a Blackwell (SM100) GPU
# — point BEAKER_CLUSTER / --gpu-type at a B200 cluster for meaningful numbers.

BENCH_DIR="src/scripts/benchmarks"

# ----------------------------------------------------------------------------- registry
# Each entry: "name|gpus|imgclass|extra args"
#   imgclass: std  -> BEAKER_IMAGE   |   symm -> BEAKER_SYMM_IMAGE (needs the ext toolchain)
BENCHES=(
  "grouped_linear_vs_grouped_mm_vs_grouped_gemm_bench|1|std|"
  "mxfp8_q_swizzle_bench|1|std|"
  "mxfp8_scale_mode_bench|1|std|"
  "mxfp8_weighted_q_bench|1|std|"
  "odc_bench|1|std|"
  "restore_unpermute_fused_bench|1|std|"
  "routed_experts_fp8_bench|1|std|"
  "scaled_grouped_mm_q_bench|1|std|"
  "shared_experts_dense_bench|1|std|"
  "swiglu_valid_prefix_bench|1|std|"
  "rowwise_alltoall_bench|8|symm|"
  "rowwise_combine_fused_bench|8|symm|"
  "rowwise_fp8_comm_bench|8|symm|"
  "all_reduce_bench|8|std|"
)

# ----------------------------------------------------------------------------- launch
launch_bench() {
  local name="$1" gpus="$2" imgclass="$3" extra="$4"
  local run_name="${RUN_PREFIX}-${name//_/-}"

  local image=""
  if [[ "${imgclass}" == "symm" ]]; then
    if [[ -z "${BEAKER_SYMM_IMAGE}" ]]; then
      echo "SKIP ${name}: set BEAKER_SYMM_IMAGE (needs the symm_mem_vdev2d ext toolchain)" >&2
      return 0
    fi
    image="${BEAKER_SYMM_IMAGE}"
  else
    image="${BEAKER_IMAGE}"
  fi

  local args=(
    --name "${run_name}"
    --task-name "${name}"
    --gpus "${gpus}"
    --nodes 1
    --budget "${BEAKER_BUDGET}"
    --workspace "${BEAKER_WORKSPACE}"
    --cluster "${BEAKER_CLUSTER}"
    --priority "${BEAKER_PRIORITY}"
    --weka "${BEAKER_WEKA}"
    --shared-filesystem
    --preemptible
    --allow-dirty
  )
  [[ -n "${image}" ]] && args+=(--beaker-image "${image}")
  if [[ ${#BEAKER_ENV_SECRETS[@]} -gt 0 && -n "${BEAKER_ENV_SECRETS[0]}" ]]; then
    args+=(--env-secret "${BEAKER_ENV_SECRETS[@]}")
  fi
  [[ "${DRY_RUN}" == "1" ]] && args+=(--dry-run)

  echo "==> launching ${name} (gpus=${gpus}, image=${image:-<default>})"
  # shellcheck disable=SC2086  # $extra is intentionally word-split into args
  python -m olmo_core.launch.beaker "${args[@]}" -- "${BENCH_DIR}/${name}.py" ${extra}
}

# ----------------------------------------------------------------------------- main
selected=("$@")
for entry in "${BENCHES[@]}"; do
  IFS='|' read -r name gpus imgclass extra <<< "${entry}"
  if [[ ${#selected[@]} -gt 0 ]]; then
    match=0
    for s in "${selected[@]}"; do [[ "${s}" == "${name}" ]] && match=1; done
    [[ "${match}" == "0" ]] && continue
  fi
  launch_bench "${name}" "${gpus}" "${imgclass}" "${extra}"
done
