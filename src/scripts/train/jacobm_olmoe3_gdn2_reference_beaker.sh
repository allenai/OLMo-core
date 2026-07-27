#!/usr/bin/env bash

set -euo pipefail

SOURCE_REPO=${SOURCE_REPO:-/weka/oe-adapt-default/jacobm/olmoe3/olmo-ddp-migration/OLMo-core-moe-v2-core}
FLA_OVERLAY=${GDN2_FLA_OVERLAY:-/tmp/fla-gdn2-cbb0a72}
FLA_SPEC=${GDN2_FLA_SPEC:-'flash-linear-attention[cuda] @ git+https://github.com/fla-org/flash-linear-attention.git@cbb0a72efb55c18ca0ef4f298298317573ad2cb3'}
FLA_EXPECTED_COMMIT=${GDN2_FLA_EXPECTED_COMMIT:-cbb0a72efb55c18ca0ef4f298298317573ad2cb3}

rm -rf "${FLA_OVERLAY}"
python -m pip install \
  --target "${FLA_OVERLAY}" \
  --no-deps \
  --no-build-isolation \
  "${FLA_SPEC}"

export PYTHONPATH="${FLA_OVERLAY}:${SOURCE_REPO}/src"
export PYTHONUNBUFFERED=1
export FLA_OVERLAY FLA_EXPECTED_COMMIT
python - <<'PY'
import json
import os
from importlib.metadata import distributions

import fla
from fla.ops.gdn2 import chunk_gdn2, naive_recurrent_gdn2

dist = next(
    dist
    for dist in distributions(path=[os.environ["FLA_OVERLAY"]])
    if dist.metadata["Name"] == "flash-linear-attention"
)
direct_url = json.loads(dist.read_text("direct_url.json"))
commit = direct_url["vcs_info"]["commit_id"]
print(f"Pinned GDN2 FLA overlay: version={fla.__version__} commit={commit} path={fla.__file__}")
print(f"GDN2 kernel: {chunk_gdn2.__module__}.{chunk_gdn2.__name__}")
print(f"GDN2 reference: {naive_recurrent_gdn2.__module__}.{naive_recurrent_gdn2.__name__}")
assert fla.__version__ == "0.5.2"
assert commit == os.environ["FLA_EXPECTED_COMMIT"]
PY

exec python \
  "${SOURCE_REPO}/src/scripts/train/jacobm_olmoe_ladder/v2/models/validate_gdn2_reference.py"
