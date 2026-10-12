#!/usr/bin/env bash
# chunk_matrix.py with the FLA autotune freeze on, then off, on the pushed current branch.
set -euo pipefail
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$HOME/.local/bin:$PATH
CMD="set -uo pipefail
echo '##### freeze ON'; python debug/gdn_int32_boundary/chunk_matrix.py 2>&1 | grep -E 'FAILED|passed|failed|^/.*Error|assert' | cut -c1-300
echo '##### freeze OFF'; python debug/gdn_int32_boundary/chunk_matrix.py nofreeze 2>&1 | grep -E 'FAILED|passed|failed|^/.*Error|assert' | cut -c1-300"
gantry run --name "${1:-chunked-prefill-matrix}" -w ai2/flex2 -b ai2/oe-other --cluster ai2/jupiter-cirrascale-2 \
  --gpus 1 --priority urgent --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
  --python-manager conda --system-python --install "pip install -e '.[fla]' pytest dataclass-extensions" \
  --timeout 0 --shared-memory 32GiB --yes -- bash -c "$CMD"
