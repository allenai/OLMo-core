#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../../../../../../.." && pwd)

cd "${REPO_ROOT}"
exec uv run python \
  src/scripts/train/jacobm_olmoe_ladder/v2/launchers/pretraining/launch_geometry_matched_scale_full.py \
  --manifest \
  src/scripts/train/jacobm_olmoe_ladder/v2/launchers/pretraining/manifests/1p2b_kda_latent_moe_l2_balanced_allocated.yaml \
  --record \
  src/scripts/train/jacobm_olmoe_ladder/v2/launchers/pretraining/generated/1p2b_kda_latent_moe_l2_balanced_allocated_submissions.json \
  "$@"
