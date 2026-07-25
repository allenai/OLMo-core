#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

exec beaker experiment create \
  "${SCRIPT_DIR}/specs/275m_gdn2_reference_validation.yaml" \
  --workspace ai2/OLMo-3-moe-experiments \
  "$@"
