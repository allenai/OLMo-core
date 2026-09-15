#!/bin/bash
# Runs INSIDE the gantry container (see build_fast2k_data_beaker.sh, which submits it). Kept as its
# own file rather than a heredoc so there is no three-level shell quoting to get wrong.
# Env: TASK BUDGETS SEQ_LEN WEKA TOKENIZER EVAL_RUNG CONV_TASK CHUNK_BY CTC_BRANCH
set -uo pipefail
export PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
PYB=/opt/conda/bin/python
echo "python: $PYB  task=$TASK budgets=$BUDGETS seq_len=$SEQ_LEN"
$PYB -m pip install -q -e . 2>&1 | tail -2
$PYB -c "import numpy, transformers, torch, olmo_core" || { echo "!!! deps missing"; exit 1; }
F=$WEKA/ds64/fast2k
mkdir -p $F/arms $F/shards
POOL=$WEKA/ds64/build/$TASK/pools/${TASK}_2k/$TASK/train.jsonl
if [ ! -s "$POOL" ]; then
  echo "--- ds64 2k pool missing at $POOL; building one with ctc-data $(date +%T) ---"
  git fetch -q origin "$CTC_BRANCH" && git checkout -q "origin/$CTC_BRANCH" -- ctc || { echo "!!! ctc checkout FAILED"; exit 1; }
  $PYB -m pip install -q -e ./ctc 2>&1 | tail -2
  $PYB -m ctc.data.cli build --task "$TASK" --out "$F/pool_$TASK" --split train --rungs 2k --train 6000 --seed 9137 --pool auto --force || exit 1
  POOL=$F/pool_$TASK/$TASK/train.jsonl
fi
echo "pool: $POOL ($(wc -l < "$POOL") rows)"
$PYB debug/ds64_fast2k/compose_fast2k.py --task "$TASK" --pool "$POOL" --out-dir "$F/arms" --budgets "$BUDGETS" || exit 1
for f in $F/arms/${TASK}_f*.jsonl; do
  ARM=$(basename "$f" .jsonl); OUT=$F/shards/$ARM
  if [ -s "$OUT/metadata.json" ]; then
    echo "[skip] shard $ARM already built"
  else
    echo "--- tokenizing $ARM $(date +%T) ---"; mkdir -p "$OUT"
    PYTHONPATH=src $PYB src/scripts/data/convert_unified_to_document_landmark.py \
      --input-jsonl "$f" --task "$CONV_TASK" --out-dir "$OUT" --emit dense --marker-set qwen3_5 \
      --tokenizer "$TOKENIZER" --seq-len "$SEQ_LEN" --query-position after --cot-mode none \
      --chunk-by "$CHUNK_BY" --emit-gold-sidecar --num-proc 16 || { echo "!!! tokenize FAILED $ARM"; exit 1; }
  fi
  echo "--- stats $ARM ---"
  $PYB debug/ds64_fast2k/fast2k_stats.py --arm-jsonl "$f" --shard-dir "$OUT" \
    --eval-jsonl "$EVAL_RUNG" --out "$OUT/fast2k_stats.json" || exit 1
done
ls -la $F/shards
echo "=== DONE $TASK ==="
