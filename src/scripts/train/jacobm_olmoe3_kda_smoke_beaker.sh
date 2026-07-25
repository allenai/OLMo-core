#!/usr/bin/env bash

set -euo pipefail

SOURCE_REPO=${SOURCE_REPO:-/weka/oe-adapt-default/jacobm/olmoe3/olmo-ddp-migration/OLMo-core-moe-v2-core}

python "${SOURCE_REPO}/src/scripts/train/jacobm_olmoe_ladder/v2/models/validate_kda_reference.py"
exec bash "${SOURCE_REPO}/src/scripts/train/jacobm_olmoe3_kda_beaker.sh"
