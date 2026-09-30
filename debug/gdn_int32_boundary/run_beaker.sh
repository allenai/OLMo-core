#!/usr/bin/env bash
# One single-GPU Beaker job on the PUSHED current branch: the GDN length-boundary probe + the
# chunked-prefill / conv GPU tests.
set -euo pipefail
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$HOME/.local/bin:$PATH
CMD="set -uo pipefail
python debug/gdn_int32_boundary/probe.py
python -m pytest -q -p no:cacheprovider src/test/generate/generation_module/transformer/generation_module_test.py -k chunk 2>&1 | tail -15"
gantry run --name "${1:-gdn-int32-boundary-probe}" -w ai2/flex2 -b ai2/oe-other --cluster ai2/jupiter-cirrascale-2 \
  --gpus 1 --priority urgent --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
  --python-manager conda --system-python --install "pip install -e '.[fla]' pytest dataclass-extensions" \
  --timeout 0 --shared-memory 32GiB --yes -- bash -c "$CMD"
