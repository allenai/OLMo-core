#!/usr/bin/env bash

set -euo pipefail

SOURCE_REPO=${SOURCE_REPO:-/weka/oe-adapt-default/jacobm/olmoe3/olmo-ddp-migration/OLMo-core-moe-v2-core}
FLA_OVERLAY=/tmp/fla-gdn2-cbb0a72
FLA_SPEC='flash-linear-attention[cuda] @ git+https://github.com/fla-org/flash-linear-attention.git@cbb0a72efb55c18ca0ef4f298298317573ad2cb3'

rm -rf "${FLA_OVERLAY}"
python -m pip install \
  --target "${FLA_OVERLAY}" \
  --no-deps \
  --no-build-isolation \
  "${FLA_SPEC}"

PYTHONPATH="${FLA_OVERLAY}" python - <<'PY'
import fla
from fla.ops.gdn2 import chunk_gdn2

print(f"Pinned GDN2 FLA overlay: version={fla.__version__} path={fla.__file__}")
print(f"GDN2 kernel: {chunk_gdn2.__module__}.{chunk_gdn2.__name__}")
assert fla.__version__ == "0.5.2"
PY

export OLMOE3_HYBRID_FLA_OVERLAY="${FLA_OVERLAY}"
exec bash "${SOURCE_REPO}/src/scripts/train/jacobm_olmoe3_hybrid_scale_beaker.sh"
