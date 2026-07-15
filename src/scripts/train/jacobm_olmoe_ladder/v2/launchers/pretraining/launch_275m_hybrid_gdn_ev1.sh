#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)
PYTHON=${PYTHON:-"${REPO_ROOT}/.venv/bin/python"}

exec "${PYTHON}" "${SCRIPT_DIR}/launch_sweep.py" \
  "${SCRIPT_DIR}/manifests/275m_hybrid_gdn_ev1.yaml" "$@"
