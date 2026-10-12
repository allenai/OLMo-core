#!/usr/bin/env bash
# chunk4_diag.py (chunk_size=4 divergence: bug or bf16 near-tie?) on the pushed current branch.
set -euo pipefail
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$HOME/.local/bin:$PATH
CMD="set -uo pipefail
python debug/gdn_int32_boundary/chunk4_diag.py 2>&1 | grep -E 'chunk=|Error' | cut -c1-260"
gantry run --name "${1:-chunked-prefill-chunk4-diag}" -w ai2/flex2 -b ai2/oe-other --cluster ai2/jupiter-cirrascale-2 \
  --gpus 1 --priority urgent --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
  --python-manager conda --system-python --install "pip install -e '.[fla]' pytest dataclass-extensions" \
  --timeout 0 --shared-memory 32GiB --yes -- bash -c "$CMD"
