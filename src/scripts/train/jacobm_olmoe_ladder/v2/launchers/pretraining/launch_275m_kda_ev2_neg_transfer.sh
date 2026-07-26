#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)
PYTHON=${PYTHON:-"${REPO_ROOT}/.venv/bin/python"}

exec "${PYTHON}" "${SCRIPT_DIR}/launch_hybrid_scale_smokes.py" \
  --manifest "${SCRIPT_DIR}/manifests/275m_kda_ev2_neg_nope_gated_transfer.yaml" \
  --output "${SCRIPT_DIR}/generated/275m_kda_ev2_neg_nope_gated_transfer.yaml" \
  --record "${SCRIPT_DIR}/generated/275m_kda_ev2_neg_nope_gated_transfer_submissions.json" \
  "$@"
