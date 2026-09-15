#!/bin/bash
# Runs INSIDE the gantry container (see build_fast8k_data_beaker.sh, which submits it). Kept as its
# own file rather than a heredoc so there is no three-level shell quoting to get wrong.
# Env: TASK BUDGETS SLICES SEQ_LEN WEKA TOKENIZER EVAL_RUNG EVAL_RUNG_2K CONV_TASK CHUNK_BY CTC_BRANCH POOL_ROWS
set -uo pipefail
export PYTHONWARNINGS=ignore TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1
PYB=/opt/conda/bin/python
echo "python: $PYB  task=$TASK budgets=$BUDGETS slices=$SLICES seq_len=$SEQ_LEN"
$PYB -m pip install -q -e . 2>&1 | tail -2
$PYB -c "import numpy, transformers, torch, olmo_core" || { echo "!!! deps missing"; exit 1; }
F=$WEKA/ds64/fast8k
mkdir -p $F/arms $F/shards
# The ds64 campaign's own 8k rung pool -- the SAME generator and seed pool the short-heavy mix's 8k
# slice is drawn from, so a fast8k row is an 8k ds64 row.
POOL=$WEKA/ds64/build/$TASK/pools/${TASK}_8k/$TASK/train.jsonl
if [ ! -s "$POOL" ]; then
  echo "--- ds64 8k pool missing at $POOL; building one with ctc-data $(date +%T) ---"
  git fetch -q origin "$CTC_BRANCH" && git checkout -q "origin/$CTC_BRANCH" -- ctc || { echo "!!! ctc checkout FAILED"; exit 1; }
  $PYB -m pip install -q -e ./ctc 2>&1 | tail -2
  $PYB -m ctc.data.cli build --task "$TASK" --out "$F/pool_$TASK" --split train --rungs 8k --train "$POOL_ROWS" --seed 9317 --pool auto --force || exit 1
  POOL=$F/pool_$TASK/$TASK/train.jsonl
fi
echo "pool: $POOL ($(wc -l < "$POOL") rows)"
$PYB debug/ds64_fast8k/compose_fast8k.py --task "$TASK" --pool "$POOL" --out-dir "$F/arms" \
  --budgets "$BUDGETS" --slices "$SLICES" || exit 1
for f in $F/arms/${TASK}_g*.jsonl; do
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
  # Two overlap checks, not one: fast8k trains on 8k rows and scores the 8k rung (the decisive
  # number) AND the 2k rung (continuity with the fast2k screen), so both have to be disjoint.
  echo "--- stats $ARM (vs the 8k eval rung) ---"
  $PYB debug/ds64_fast2k/fast2k_stats.py --arm-jsonl "$f" --shard-dir "$OUT" \
    --eval-jsonl "$EVAL_RUNG" --out "$OUT/fast8k_stats.json" || exit 1
  echo "--- stats $ARM (vs the 2k eval rung) ---"
  $PYB debug/ds64_fast2k/fast2k_stats.py --arm-jsonl "$f" --shard-dir "$OUT" \
    --eval-jsonl "$EVAL_RUNG_2K" --out "$OUT/fast8k_stats_2k.json" || exit 1
done
ls -la $F/shards
echo "=== DONE $TASK ==="
