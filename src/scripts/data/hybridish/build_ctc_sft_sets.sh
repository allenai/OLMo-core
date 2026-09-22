#!/bin/bash
# The two CTC SFT training sets, built TOKEN-BALANCED across context buckets.
#
#   bash src/scripts/data/hybridish/build_ctc_sft_sets.sh <set-a|set-b> <out-root>
#
# TOKEN-BALANCED, not example-balanced (prasann's call). Every context bucket gets the SAME TOKEN
# BUDGET, so example count falls as the bucket grows: at a 20M-token budget a 2k bucket holds ~9,760
# examples and a 256k bucket holds ~76. The alternative -- equal examples per bucket -- spends ~99%
# of the token budget above 32k and trains the short buckets on almost nothing.
#
# query_position=both throughout. NOTE it is NOT a `ctc-data build` flag -- the rung files are raw
# unified JSONL and the prompt is rendered later, so `both` is passed to
# convert_ctc_to_sft_completion.py at the tokenisation step (see the tail of this script). It must
# agree with the EVAL flag: scoring a query-after model with `both` hands it a second copy of the ask
# it never saw in training, which reads as a capability gap rather than a prompt mismatch.
#
# ⚠ TWO ROSTER FACTS THAT DECIDE WHAT THESE SETS MEASURE:
#  * `fiqa`, `scifact`, `outlier_review` and `contra_fever` are NEVER trained -- build_ctc_sft_mix.py
#    refuses them outright. They are the OOD columns; training one makes its number meaningless.
#  * Both sets collide on a grading spec and so need --allow-spec-collision, which the manifest
#    records. SET-B trains nq+hotpotqa (both `retrieval`) and outlier+outlier_amzn (both `outlier`).
#    ⚠ outlier_amzn is HALF category-axis Amazon reviews and outlier_review IS category-axis Amazon
#    reviews, so SET-B weakens the outlier_review probe specifically. Swap outlier_amzn for
#    outlier_fixedM if that probe matters more than matching CTC-BENCH-10 exactly.
set -uo pipefail
SET="${1:?usage: build_ctc_sft_sets.sh <set-a|set-b> <out-root>}"
ROOT="${2:?}"
PY="${PY:-/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python}"
# NOT the `ctc-data` console script: it is shebanged to a python with no huggingface_hub, and
# `--pool auto` fetches the seed pools from the Hub, so every build dies at the first task with
# ModuleNotFoundError. Drive the CLI as a module under an interpreter that has the Hub client.
CTC_SRC="${CTC_SRC:-/accounts/projects/berkeleynlp/prasann/projects/newolmocore/OLMo-core/ctc/src}"
CTC_DATA="${CTC_DATA:-$PY -m ctc.data.cli}"
export PYTHONPATH="$CTC_SRC:${PYTHONPATH:-}"
BUCKETS="${BUCKETS:-2k 4k 8k 16k 32k}"
TOK_PER_BUCKET="${TOK_PER_BUCKET:-20000000}"   # 20M tokens in EVERY bucket
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core

# SET-A: as many of the 22 as are buildable, minus the held-out four. Trains 5 of 7 retrieval
# sources and all 3 qdmatch, so the surviving OOD columns measure CORPUS transfer only, not
# unseen-task transfer -- say so wherever those numbers appear.
# ⚠ Only what `ctc-data build` can actually produce. Six roster members have NO ctc-data generator
# -- msmarco, niah, obliq_twitter, qdmatch_fiqa, outlier_amzn, outlier_fixedM -- so "as many of the
# 22 as possible" is FOURTEEN here, not twenty. The missing six need their own generators
# (outlier_amzn/outlier_fixedM: generate_review_outlier_data.py; the rest: see BUILD_MATRIX.md).
# qdmatch_hpqa dropped (prasann): qdmatch is then trained from qdmatch_nq ALONE, which keeps
# qdmatch_hpqa and qdmatch_fiqa as clean held-out probes for that spec. Without it the qdmatch
# column would have had nothing left to generalise to.
SET_A="nq hotpotqa qdmatch_nq outlier oolong contradiction xabsence absence \
       reorder rerank strmatch textgroups grouping_labeled"
# SET-B: CTC-BENCH-10 with prasann's substitutions -- rerank for fiqa, reorder for qdmatch_fiqa.
# Both swaps exist to keep FiQA out of training entirely: fiqa is a held-out probe and qdmatch_fiqa
# is built from the same corpus, so training it would contaminate that probe.
# ⚠ outlier_amzn has no ctc-data generator either. Until it is built with
# generate_review_outlier_data.py (--rating-ratio 0.5, the shipped blend), SET-B is NINE tasks and
# the "Outlier (Amazon)" row of CTC-BENCH-10 is missing from training.
SET_B="nq hotpotqa rerank oolong reorder qdmatch_nq outlier xabsence contradiction"

case "$SET" in
  set-a) TASKS="$SET_A"; TAG=setA_max20 ;;
  set-b) TASKS="$SET_B"; TAG=setB_bench10 ;;
  *) echo "unknown set '$SET' (want set-a or set-b)"; exit 2 ;;
esac

BUILD="$ROOT/$TAG/per_task"
mkdir -p "$BUILD"
echo "=== $TAG | $(echo $TASKS | wc -w) tasks | buckets: $BUCKETS | ${TOK_PER_BUCKET} tok/bucket ==="

for task in $TASKS; do
  for b in $BUCKETS; do
    # examples = budget / bucket size, so every bucket costs the same tokens
    case "$b" in
      2k) TOK=2048;; 4k) TOK=4096;; 8k) TOK=8192;; 16k) TOK=16384;; 32k) TOK=32768;;
      64k) TOK=65536;; 128k) TOK=131072;; 256k) TOK=262144;;
      *) echo "unknown bucket $b"; exit 2;;
    esac
    N=$(( TOK_PER_BUCKET / TOK ))
    echo "--- $task @ $b : $N examples (~${TOK_PER_BUCKET} tok) $(date -u '+%T')Z ---"
    # EACH BUCKET GETS ITS OWN --out. `ctc-data build` always writes <out>/<task>/train.jsonl, so
    # reusing one --out across buckets silently CLOBBERS each bucket with the next: the tree ends
    # up holding only the LAST (smallest) bucket -- the exact opposite of token-balanced, and it
    # looks like a clean successful build while doing it. Merged per task immediately below.
    $CTC_DATA build --task "$task" --split train --rungs "$b" --train "$N" \
      --pool auto --out "$BUILD/_b$b" \
      || echo "  !!! $task@$b FAILED (continuing; the mix step reports what is missing)"
  done
  mkdir -p "$BUILD/$task"
  : > "$BUILD/$task/train.jsonl"
  for b in $BUCKETS; do
    f="$BUILD/_b$b/$task/train.jsonl"
    [ -s "$f" ] && cat "$f" >> "$BUILD/$task/train.jsonl"
  done
  echo "=== $task merged: $(wc -l < "$BUILD/$task/train.jsonl") rows over $(echo $BUCKETS | wc -w) buckets ==="
done

echo "=== assembling the mix ==="
$PY "$REPO/src/scripts/data/hybridish/build_ctc_sft_mix.py" \
  --root "$BUILD" --tasks $TASKS --allow-spec-collision \
  --band "$(echo $BUCKETS | tr ' ' '-')" \
  --out "$ROOT/$TAG/mix.jsonl"
echo "=== tokenise (query_position=both is applied HERE, not at build time) ==="
echo "  $PY $REPO/src/scripts/data/hybridish/convert_ctc_to_sft_completion.py \\"
echo "      --mix $ROOT/$TAG/mix.jsonl --query-position both --verify --out $ROOT/$TAG/shards"
echo "=== DONE $TAG -> $ROOT/$TAG/mix.jsonl ==="
