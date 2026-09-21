#!/bin/bash
# outlier_review as a CTC-suite row: 2k-256k, native regeneration.
#
# WHY REGENERATE INSTEAD OF EXPANDING. outlier gold is STRUCTURAL ("the documents from the least
# common category"), so injecting fillers makes injected documents satisfy the gold condition
# without being labelled -- the failure that took the old outlier xlong ladder from 0.428 @32k to
# 0.069 @65k. expand_ctc_rung refuses the task for exactly this reason. The generator recomputes
# gold at the larger n, so it is the only sound route.
#
# WHAT THIS ROW IS, PRECISELY. Same generator and corpus as the suite's `outlier_amzn` row, but
# pinned to the CATEGORY axis (`--rating-ratio 0`): the majority of reviews share a product
# category and the outliers come from another one. That is the domain-shift control for a model
# trained on wiki-category `outlier`. It is NOT redundant with outlier_amzn, which is a 50/50 blend
# of this and a star-RATING axis -- a different question over the same reviews.
#
# n per rung mirrors outlier_amzn's grid so the two Amazon rows share an x-axis. K=3 matches the
# historical outlier_review probe. eval_size 500 to 128k, 125 at 256k (suite policy).
#
#   bash debug/ctc_fast_suite/build_outlier_review_ladder.sh
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
GEN=$REPO/src/corpus_reasoning/data/generate_review_outlier_data.py
OUT=${OUT:-/scratch/users/prasann/ctc_ood_ladders/outlier_review}
export PYTHONPATH=$REPO/src
export HF_HOME=/scratch/users/prasann/hf_cache
export HF_DATASETS_CACHE=$HF_HOME
export HF_HUB_CACHE=$HF_HOME/hub
mkdir -p "$OUT"

echo "=== outlier_review ladder START=$(date -u '+%F %T')Z ==="
# rung_label:num_docs:eval_size
for spec in 2048:20:500 4096:40:500 8192:80:500 16384:160:500 32768:320:500 \
            65536:596:500 131072:1207:500 262144:2429:125; do
  LAB="${spec%%:*}"; REST="${spec#*:}"; N="${REST%%:*}"; SZ="${REST##*:}"
  DST="$OUT/rung_${LAB}.jsonl"
  if [ -s "$DST" ]; then echo "  [skip] rung_${LAB} exists"; continue; fi
  echo "######## rung_${LAB}  n=$N docs, K=3, category-only, ${SZ} rows  $(date -u '+%T')Z ########"
  RAW="$OUT/_raw_review_outlier_category_n${N}.jsonl"
  $PY -u "$GEN" --num-examples "$SZ" --num-docs "$N" --num-outliers 3 \
      --rating-ratio 0.0 --seed 7 --pool-size 20000 --out "$RAW" \
      2>&1 | tail -12 || { echo "  !!! rung_${LAB} FAILED"; continue; }
  head -"$SZ" "$RAW" > "$DST"
  echo "  wrote $(wc -l < "$DST") rows -> $DST"
done
echo "=== DONE $(date -u '+%F %T')Z ==="
