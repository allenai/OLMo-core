#!/usr/bin/env bash
set -euo pipefail

global_rank="${RANK:-0}"
rank_stride="${NSYS_PROFILE_RANK_STRIDE:-8}"

if (( global_rank % rank_stride == 0 )); then
    output_dir="${BEAKER_RESULT_DIR:-/results}/nsight"
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
