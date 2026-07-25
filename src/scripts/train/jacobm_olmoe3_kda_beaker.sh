#!/usr/bin/env bash

set -euo pipefail

SOURCE_REPO=${SOURCE_REPO:-/weka/oe-adapt-default/jacobm/olmoe3/olmo-ddp-migration/OLMo-core-moe-v2-core}

python - <<'PY'
import fla
from fla.ops.kda import chunk_kda

print(f"KDA FLA environment: version={fla.__version__} path={fla.__file__}")
print(f"KDA training kernel: {chunk_kda.__module__}.{chunk_kda.__name__}")
assert fla.__version__ == "0.4.1"
PY

exec bash "${SOURCE_REPO}/src/scripts/train/jacobm_olmoe3_hybrid_scale_beaker.sh"
