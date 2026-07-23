#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)
PYTHON=${PYTHON:-"${REPO_ROOT}/.venv/bin/python"}

args=(
  --manifest "${SCRIPT_DIR}/manifests/275m_rope_gated_bucket_sweep.yaml"
  --output "${SCRIPT_DIR}/generated/275m_rope_gated_bucket_sweep.yaml"
  --record "${SCRIPT_DIR}/generated/275m_rope_gated_bucket_sweep_submissions.json"
  --experiment-name jacobm-moe-v2-core-275m-rope-gated-bucket-sweep-r1
)

if [[ ${1:-} == "--submit" ]]; then
  args+=(--submit)
  shift
fi

exec "${PYTHON}" "${SCRIPT_DIR}/launch_275m_parallelism_smokes.py" "${args[@]}" "$@"
