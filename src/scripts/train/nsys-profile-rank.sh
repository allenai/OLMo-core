#!/usr/bin/env bash
set -euo pipefail

global_rank="${RANK:-0}"
rank_stride="${NSYS_PROFILE_RANK_STRIDE:-8}"
experiment_id="${BEAKER_EXPERIMENT_ID:-local}"

if [[ -n "${TORCH_FR_DUMP_TEMP_FILE:-}" ]]; then
    flight_recorder_dir="$(dirname "${TORCH_FR_DUMP_TEMP_FILE}")/${experiment_id}"
    flight_recorder_prefix="$(basename "${TORCH_FR_DUMP_TEMP_FILE}")"
    mkdir -p "${flight_recorder_dir}"
    export TORCH_FR_DUMP_TEMP_FILE="${flight_recorder_dir}/${flight_recorder_prefix}"
fi

if (( global_rank % rank_stride == 0 )); then
    output_dir="${NSYS_OUTPUT_DIR:-${BEAKER_RESULT_DIR:-/results}/nsight}"
    output_dir="${output_dir}/${experiment_id}"
    mkdir -p "${output_dir}"
    exec nsys profile \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --trace=cuda,nvtx,osrt,nccl \
        --sample=process-tree \
        --backtrace=fp \
        --force-overwrite=true \
        --output="${output_dir}/rank-${global_rank}" \
        python "$@"
else
    exec python "$@"
fi
