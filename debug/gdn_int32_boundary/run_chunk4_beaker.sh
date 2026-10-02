#!/usr/bin/env bash
# Why does test_generation_module_chunked_prefill_matches_one_shot fail at chunk_size=4 on
# prasann/landmark-chunked-prefill (5 and 16 pass)? Full tracebacks here, then the same test on
# Amanda's original branch (prasann/ctc-setA-sft) to tell a port bug from a pre-existing one.
set -euo pipefail
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$HOME/.local/bin:$PATH
T=src/test/generate/generation_module/transformer/generation_module_test.py
CMD="set -uo pipefail
echo '##### port (this branch)'
python -m pytest -q -p no:cacheprovider $T -k 'chunked_prefill and 4' -x --tb=long 2>&1 | grep -vE '^\s*$' | tail -60
echo '##### original (prasann/ctc-setA-sft)'
git fetch -q origin prasann/ctc-setA-sft && git checkout -q FETCH_HEAD -- src/olmo_core src/test
python -m pytest -q -p no:cacheprovider $T -k 'chunked_prefill' --tb=line 2>&1 | tail -12"
gantry run --name "${1:-chunked-prefill-chunk4}" -w ai2/flex2 -b ai2/oe-other --cluster ai2/jupiter-cirrascale-2 \
  --gpus 1 --priority urgent --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 \
  --python-manager conda --system-python --install "pip install -e '.[fla]' pytest dataclass-extensions" \
  --timeout 0 --shared-memory 32GiB --yes -- bash -c "$CMD"
