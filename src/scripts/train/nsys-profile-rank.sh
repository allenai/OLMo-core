#!/usr/bin/env bash
set -euo pipefail

global_rank="${RANK:-0}"
rank_stride="${NSYS_PROFILE_RANK_STRIDE:-8}"
experiment_id="${BEAKER_EXPERIMENT_ID:-local}"

ensure_nsys() {
    if command -v nsys >/dev/null 2>&1; then
        return
    fi

    if (( EUID != 0 )); then
        echo "nsys is not installed and cannot be installed without root" >&2
        return 1
    fi

    echo "nsys is not installed; installing pinned Nsight Systems CLI 2025.5.1" >&2
    # Keep this in sync with the pinned Nsight installation in src/Dockerfile.
    . /etc/os-release
    ubuntu_short="$(printf '%s' "${VERSION_ID}" | tr -d .)"
    arch="$(dpkg --print-architecture)"
    printf \
        'deb [trusted=yes] https://developer.download.nvidia.com/devtools/repos/ubuntu%s/%s/ /\n' \
        "${ubuntu_short}" "${arch}" \
        > /etc/apt/sources.list.d/nvidia-devtools.list
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        nsight-systems-cli-2025.5.1
}

if [[ -n "${TORCH_FR_DUMP_TEMP_FILE:-}" ]]; then
    flight_recorder_dir="$(dirname "${TORCH_FR_DUMP_TEMP_FILE}")/${experiment_id}"
    flight_recorder_prefix="$(basename "${TORCH_FR_DUMP_TEMP_FILE}")"
    mkdir -p "${flight_recorder_dir}"
    export TORCH_FR_DUMP_TEMP_FILE="${flight_recorder_dir}/${flight_recorder_prefix}"
fi

if (( global_rank % rank_stride == 0 )); then
    # The B300 training image used by this reproduction predates the Nsight
    # layer in src/Dockerfile. Only one rank per node enters this branch, so a
    # missing package is installed exactly once on each node.
    ensure_nsys

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
