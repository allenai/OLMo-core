#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

exec beaker experiment create \
  "${SCRIPT_DIR}/specs/matched_kda_gdn2_numerics.yaml" \
  --workspace ai2/OLMo-3-moe-experiments \
  "$@"
