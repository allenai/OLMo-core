#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-python}

args=(
  --manifest "${SCRIPT_DIR}/manifests/275m_rope_gated_large_batch_parallelism.yaml"
  --output "${SCRIPT_DIR}/generated/275m_rope_gated_large_batch_parallelism.yaml"
  --record "${SCRIPT_DIR}/generated/275m_rope_gated_large_batch_parallelism_submissions.json"
  --experiment-name jacobm-moe-v2-core-275m-large-batch-parallelism-r1
)

if [[ ${1:-} == "--submit" ]]; then
  args+=(--submit)
  shift
fi

exec "${PYTHON}" "${SCRIPT_DIR}/launch_275m_parallelism_smokes.py" "${args[@]}" "$@"
