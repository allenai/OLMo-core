#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PYTHON=${PYTHON:-python}

exec "${PYTHON}" "${SCRIPT_DIR}/launch_hybrid_scale_smokes.py" \
  --manifest "${SCRIPT_DIR}/manifests/480m_kda_latent_moe_l2_holmes_cx1.yaml" \
  --output "${SCRIPT_DIR}/generated/480m_kda_latent_moe_l2_holmes_cx1.yaml" \
  --record "${SCRIPT_DIR}/generated/480m_kda_latent_moe_l2_holmes_cx1_submissions.json" \
  "$@"
