#!/bin/bash
# Build the 4-task fast8k shard (tokdrop_lr LR ablation): ONE training job per (arm, LR) covers
# contradiction + nq + outlier + oolong. CPU-only gantry job; needs the four 16M task shards
# (build_fast8k_data_beaker.sh) and the oolong arm jsonl on weka.
#
#   bash research/token_routing/ds64_fast8k/build_mt4_beaker.sh
#
# Step 1 re-tokenizes oolong with --gold-sidecar-empty-for-missing: oolong rows have no gold docs, and
# a row MISSING from gold_fingerprints.json is a fingerprint miss at train time, which keeps every
# document real -- the token-drop arm would silently train oolong uncompressed. Step 2 merges the
# four shards into $WEKA/ds64/fast8k/shards/mt4_g64M (merge_tasks.py).
set -uo pipefail
WEKA=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
gantry run --name "fast8k-data-mt4-$(date +%m%d%H%M)" -w ai2/flex2 -b ai2/oe-other \
  --cluster ai2/phobos --cluster ai2/hammond --cluster 'ai2/jupiter*' --cluster 'ai2/ceres*' \
  --gpus 0 --cpus 4 --memory 32GiB --priority urgent \
  --beaker-image tylerr/olmo-core-tch291cu128-2025-11-25 --install false \
  --weka oe-training-default:/weka/oe-training-default \
  --env WEKA="$WEKA" \
  --allow-dirty --timeout 0 --yes -- bash -c '
set -uo pipefail
PYB=/opt/conda/bin/python
F=$WEKA/ds64/fast8k
$PYB -m pip install -q -e . 2>&1 | tail -2
OUT=$F/shards/oolong_g16M_eg
if [ -s $OUT/metadata.json ]; then echo "[skip] $OUT built"; else
  mkdir -p $OUT
  PYTHONPATH=src $PYB recipes/data/convert_unified_to_document_landmark.py \
    --input-jsonl $F/arms/oolong_g16M.jsonl --task oolong --out-dir $OUT --emit dense --marker-set qwen3_5 \
    --tokenizer $WEKA/hf_tokenizers/Qwen3.5-0.8B-Base --seq-len 24576 --query-position after --cot-mode none \
    --chunk-by line --emit-gold-sidecar --gold-sidecar-empty-for-missing --num-proc 4 || { echo "!!! oolong FAILED"; exit 1; }
fi
$PYB -c "import json;print(\"oolong_eg sidecar entries:\", len(json.load(open(\"$OUT/gold_fingerprints.json\"))))"
$PYB research/token_routing/ds64_fast8k/merge_tasks.py --out $F/shards/mt4_g64M \
  --src $F/shards/contradiction_g16M $F/shards/nq_g16M $F/shards/outlier_g16M $OUT || exit 1
echo MT4_DONE'
