#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-python}

exec "${PYTHON}" "${SCRIPT_DIR}/launch_hybrid_scale_smokes.py" \
  --manifest "${SCRIPT_DIR}/manifests/275m_kda_latent_moe_l4_1000e_lr_sweep_cx1_cx2.yaml" \
  --output "${SCRIPT_DIR}/generated/275m_kda_latent_moe_l4_1000e_lr_sweep_cx1_cx2.yaml" \
  --record "${SCRIPT_DIR}/generated/275m_kda_latent_moe_l4_1000e_lr_sweep_submissions.json" \
  "$@"
