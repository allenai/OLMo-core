#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "usage: $0 OUTPUT_PREFIX COMMAND [ARG ...]" >&2
  exit 2
fi

output_prefix=$1
shift
profile_rank=${NSYS_PROFILE_RANK:-0}

if [[ "${RANK:?RANK must be set by torchrun}" == "${profile_rank}" ]]; then
  exec nsys profile \
    --sample=none \
    --cpuctxsw=none \
    --trace="${NSYS_TRACE:-cuda,nvtx,osrt,cublas}" \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --cuda-memory-usage=true \
    --force-overwrite=true \
    --output="${output_prefix}-rank-${RANK}" \
    "$@"
fi

exec "$@"
