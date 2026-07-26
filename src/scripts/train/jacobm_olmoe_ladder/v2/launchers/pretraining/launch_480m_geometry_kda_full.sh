#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)
PYTHON=${PYTHON:-"${REPO_ROOT}/.venv/bin/python"}

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec "${PYTHON}" "${SCRIPT_DIR}/launch_geometry_matched_scale_full.py" \
  --manifest "${SCRIPT_DIR}/manifests/480m_geometry_kda_ev1_noneg_nope_gated.yaml" \
  --record "${SCRIPT_DIR}/generated/480m_geometry_kda_ev1_noneg_nope_gated_submissions.json" \
  "$@"
