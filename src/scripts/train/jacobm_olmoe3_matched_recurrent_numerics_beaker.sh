#!/usr/bin/env bash

set -euo pipefail

SOURCE_REPO=${SOURCE_REPO:-/weka/oe-adapt-default/jacobm/olmoe3/olmo-ddp-migration/OLMo-core-moe-v2-core}
VALIDATOR="${SOURCE_REPO}/src/scripts/train/jacobm_olmoe_ladder/v2/models/validate_matched_recurrent_numerics.py"
FLA_OVERLAY=/tmp/fla-gdn2-cbb0a72
FLA_SPEC='flash-linear-attention[cuda] @ git+https://github.com/fla-org/flash-linear-attention.git@cbb0a72efb55c18ca0ef4f298298317573ad2cb3'

export PYTHONUNBUFFERED=1

PYTHONPATH="${SOURCE_REPO}/src" python - <<'PY'
import fla
assert fla.__version__ == "0.4.1", fla.__version__
print(f"Matched audit KDA environment: version={fla.__version__} path={fla.__file__}")
PY
PYTHONPATH="${SOURCE_REPO}/src" python "${VALIDATOR}" \
  --mixer kda \
  --output /results/kda_matched_numerics.json

rm -rf "${FLA_OVERLAY}"
python -m pip install \
  --target "${FLA_OVERLAY}" \
  --no-deps \
  --no-build-isolation \
  "${FLA_SPEC}"

PYTHONPATH="${FLA_OVERLAY}:${SOURCE_REPO}/src" python - <<'PY'
import fla
assert fla.__version__ == "0.5.2", fla.__version__
print(f"Matched audit GDN2 environment: version={fla.__version__} path={fla.__file__}")
PY
PYTHONPATH="${FLA_OVERLAY}:${SOURCE_REPO}/src" python "${VALIDATOR}" \
  --mixer gdn2 \
  --output /results/gdn2_matched_numerics.json

PYTHONPATH="${SOURCE_REPO}/src" python "${VALIDATOR}" \
  --compare /results/kda_matched_numerics.json /results/gdn2_matched_numerics.json \
  --markdown /results/matched_recurrent_numerics.md
