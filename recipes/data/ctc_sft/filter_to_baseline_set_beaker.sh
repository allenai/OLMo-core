#!/bin/bash
# Beaker CPU job: cut a re-rendered setA shard dir to the baseline 32k arm's examples
# (filter_to_baseline_set.py; numpy only). Defaults: the hashed-id set.
#   bash recipes/data/ctc_sft/filter_to_baseline_set_beaker.sh
set -euo pipefail
SETS=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_sft_sets
BASE="${BASE:-$SETS/setA_max20_evaliid/shards_qwen35_256k}"
NEW="${NEW:-$SETS/setA_max20_evaliid_hashid/shards_qwen35_256k}"
OUT="${OUT:-$SETS/setA_max20_evaliid_hashid/shards_qwen35_q35le32k}"
WORK="git log -1 --format='commit %H %s'; /opt/conda/bin/python -u recipes/data/ctc_sft/filter_to_baseline_set.py --baseline $BASE --rerender $NEW --out $OUT"
export PATH=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin:$PATH
gantry run --name "ctc-setA-filter-baseline-$(date +%m%d%H%M)" -w ai2/flex2 -b ai2/oe-other \
  --cluster ai2/neptune-cirrascale --cluster ai2/ceres-cirrascale --cluster ai2/saturn-cirrascale \
  --cluster ai2/jupiter-cirrascale-2 --gpus 0 --cpus 8 --memory 64GiB --priority urgent \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 --install true \
  --weka oe-training-default:/weka/oe-training-default \
  --allow-dirty --timeout 0 --yes -- bash -c "$WORK" 2>&1 | grep -E "beaker.org/ex|rror" | head -3
