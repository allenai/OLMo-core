#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)
PYTHON=${PYTHON:-"${REPO_ROOT}/.venv/bin/python"}

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TRITON_INTERPRET=${TRITON_INTERPRET:-1}
exec "${PYTHON}" "${SCRIPT_DIR}/launch_geometry_matched_scale_full.py" \
  --manifest "${SCRIPT_DIR}/manifests/480m_kda_aggressive_mxfp8_smokes.yaml" \
  --record "${SCRIPT_DIR}/generated/480m_kda_aggressive_mxfp8_smoke_submissions.json" \
  "$@"
