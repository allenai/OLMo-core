#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)
PYTHON=${PYTHON:-"${REPO_ROOT}/.venv/bin/python"}

exec "${PYTHON}" "${SCRIPT_DIR}/launch_hybrid_scale_smokes.py" \
  --manifest "${SCRIPT_DIR}/manifests/275m_gdn2_stability_ablation.yaml" \
  --output "${SCRIPT_DIR}/generated/275m_gdn2_stability_ablation.yaml" \
  --record "${SCRIPT_DIR}/generated/275m_gdn2_stability_ablation_submissions.json" \
  "$@"
