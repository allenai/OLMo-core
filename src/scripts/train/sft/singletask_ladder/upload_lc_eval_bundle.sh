#!/bin/bash
# Top-up the WEKA eval bundle that the ON-BEAKER multi-rung native eval reads from.
#
# weka is not mounted on this host, so we push through S3 (s3://ai2-llm == the weka backing bucket;
# on a Beaker node it appears at /weka/oe-training-default/ai2-llm). A prior workflow already created
# `_eval_bundle/{scripts,data}` + `_eval_bundle_eval500/` (2026-06-28). This script makes that bundle
# sufficient for `run_beaker_multirung_eval.sh` by:
#   1. refreshing the eval CODE (incl. eval_lc_native.py with the new EVAL500_ROOT env hook),
#   2. adding the per-task BASE-rung + rerank-CE data files the bundle was missing,
#   3. uploading the on-node runner script itself,
#   4. (optional, FULL=1) re-syncing the whole scripts/, data subset, and eval500 trees.
#
# Idempotent: re-running only re-uploads changed/new objects. Needs AWS creds for s3://ai2-llm.
#
#   bash src/scripts/train/sft/singletask_ladder/upload_lc_eval_bundle.sh            # top-up (fast)
#   FULL=1 bash src/scripts/train/sft/singletask_ladder/upload_lc_eval_bundle.sh     # also re-sync full scripts/ + eval500
set -euo pipefail

CR="${CR:-/scratch/users/prasann/corpus-reasoning}"
EVAL500_SRC="${EVAL500_SRC:-/scratch/users/prasann/cpt_data/eval500}"
REPO="${REPO:-/accounts/projects/berkeleynlp/prasann/projects/OLMo-core}"
S3="${S3:-s3://ai2-llm/checkpoints/prasanns}"
BUNDLE="$S3/_eval_bundle"
EVAL500_S3="$S3/_eval_bundle_eval500"

echo "=== upload eval bundle: CR=$CR -> $BUNDLE (FULL=${FULL:-0}) ==="

# 1. eval CODE (always refresh the eval scripts; FULL also re-syncs the whole scripts/ tree).
if [ "${FULL:-0}" = "1" ]; then
  echo "--- [FULL] sync scripts/ tree ---"
  aws s3 sync "$CR/scripts/" "$BUNDLE/scripts/" --exclude '*__pycache__*' --exclude '*.pyc' --size-only
else
  echo "--- refresh eval scripts (eval_lc_native + variants + evaluate + __init__) ---"
  for f in __init__.py eval_lc_native.py eval_lc_native_docchunk.py \
           eval_lc_native_docchunk_contra.py eval_lc_native_landmark.py evaluate.py; do
    [ -f "$CR/scripts/eval/$f" ] && aws s3 cp "$CR/scripts/eval/$f" "$BUNDLE/scripts/eval/$f"
  done
fi

# 2. per-task BASE-rung + rerank-CE data files the bundle was missing.
echo "--- data: base-rung + rerank-CE files ---"
for f in n2ified_eval_nq_q50.jsonl \
         msmarco_trainhn_eval_k20_500.jsonl \
         msmarco_trainhn_eval_k50_500.jsonl \
         msmarco_trainhn_eval_k100_500.jsonl \
         contradiction_eval_pubmed_both_n100_k3.jsonl; do
  if [ -f "$CR/data/$f" ]; then aws s3 cp "$CR/data/$f" "$BUNDLE/data/$f"
  else echo "WARN missing local $CR/data/$f"; fi
done
# outlier base rung (3k) reads data/outlier_wiki100w_n55_k3_eval_600.jsonl; same content lives in eval500.
OUT_N55="$EVAL500_SRC/outlier/outlier_wiki100w_n55_k3_eval_600.jsonl"
[ -f "$OUT_N55" ] && aws s3 cp "$OUT_N55" "$BUNDLE/data/outlier_wiki100w_n55_k3_eval_600.jsonl" \
  || echo "WARN missing $OUT_N55"

# 3. the on-node runner script.
echo "--- runner script ---"
aws s3 cp "$REPO/src/scripts/train/sft/singletask_ladder/run_beaker_multirung_eval.sh" "$BUNDLE/run_beaker_multirung_eval.sh"

# 4. eval500 ladder files (the _600/_500 rungs). Present since 2026-06-28; re-sync only on FULL.
if [ "${FULL:-0}" = "1" ]; then
  echo "--- [FULL] sync eval500 ladder tree ---"
  aws s3 sync "$EVAL500_SRC/" "$EVAL500_S3/" --exclude '*.log' --size-only
fi

echo "=== bundle ready ==="
echo "  code:    $BUNDLE/scripts/eval/eval_lc_native.py"
echo "  data:    $BUNDLE/data/"
echo "  eval500: $EVAL500_S3/"
echo "  runner:  $BUNDLE/run_beaker_multirung_eval.sh"
