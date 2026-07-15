#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)
PYTHON=${PYTHON:-"${REPO_ROOT}/.venv/bin/python"}

exec "${PYTHON}" "${SCRIPT_DIR}/launch_hybrid_scale_smokes.py" \
  --manifest "${SCRIPT_DIR}/manifests/hybrid_scale_full_cx1_cx2.yaml" \
  --output "${SCRIPT_DIR}/generated/hybrid_scale_full_cx1_cx2.yaml" \
  "$@"
