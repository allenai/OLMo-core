#!/bin/bash
# The CTC SFT training sets, built TOKEN-BALANCED across context buckets, in PARALLEL.
#
#   bash src/scripts/data/hybridish/build_ctc_sft_sets.sh <set-a|set-b> <out-root>
#
# The 65 (task, bucket) builds are independent, so they run under `xargs -P`. Serially this is ~6
# hours; on a wide CPU box it is ~20 minutes. NPROC defaults to the machine's core count.
#
# TOKEN-BALANCED, not example-balanced: every bucket gets the same token budget, so example count
# falls as the bucket grows (~9,765 at 2k, ~610 at 32k). Equal examples per bucket would spend ~99%
# of the budget above 32k.
#
# query_position=both is NOT a `ctc-data build` flag -- rung files are raw unified JSONL and the
# prompt is rendered at tokenisation, so it is passed to convert_ctc_to_sft_completion.py (printed
# at the end). It must match the EVAL flag or the mismatch reads as a capability gap.
#
# ⚠ EACH BUCKET NEEDS ITS OWN --out. `ctc-data build` always writes <out>/<task>/train.jsonl, so a
# shared --out silently clobbers each bucket with the next and leaves only the smallest one, looking
# like a clean build the whole way. Buckets go to _b<bucket>/ and are merged per task at the end.
#
# ⚠ NOT the `ctc-data` console script: it is shebanged to a python without huggingface_hub, and
# `--pool auto` fetches seed pools from the Hub, so every build dies at the first task.
#
# ⚠ `fiqa`, `scifact`, `outlier_review`, `contra_fever` are NEVER trained (build_ctc_sft_mix.py
# refuses them) -- they are the OOD columns. Six roster members have no ctc-data generator at all
# (msmarco, niah, obliq_twitter, qdmatch_fiqa, outlier_amzn, outlier_fixedM), which is why set-a is
# 13 tasks rather than 20.
set -uo pipefail
SET="${1:?usage: build_ctc_sft_sets.sh <set-a|set-b> <out-root>}"
ROOT="${2:?}"
PY="${PY:-/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python}"
CTC_SRC="${CTC_SRC:-/accounts/projects/berkeleynlp/prasann/projects/newolmocore/OLMo-core/ctc/src}"
REPO="${REPO:-/accounts/projects/berkeleynlp/prasann/projects/OLMo-core}"
BUCKETS="${BUCKETS:-2k 4k 8k 16k 32k}"
TOK_PER_BUCKET="${TOK_PER_BUCKET:-20000000}"
NPROC="${NPROC:-$(nproc)}"
export PYTHONPATH="$CTC_SRC:${PYTHONPATH:-}"

# qdmatch_hpqa dropped: qdmatch is then trained from qdmatch_nq ALONE, keeping qdmatch_hpqa and
# qdmatch_fiqa as clean held-out probes for that spec.
SET_A="nq hotpotqa qdmatch_nq outlier oolong contradiction xabsence absence \
       reorder rerank strmatch textgroups grouping_labeled"
# CTC-BENCH-10 with prasann's substitutions (rerank for fiqa, reorder for qdmatch_fiqa) -- both keep
# the FiQA corpus out of training so the fiqa probe stays clean. outlier_amzn has no ctc-data
# generator, so this is nine tasks until it is built with generate_review_outlier_data.py.
SET_B="nq hotpotqa rerank oolong reorder qdmatch_nq outlier xabsence contradiction"

case "$SET" in
  set-a) TASKS="$SET_A"; TAG=setA_max20 ;;
  set-b) TASKS="$SET_B"; TAG=setB_bench10 ;;
  *) echo "unknown set '$SET' (want set-a or set-b)"; exit 2 ;;
esac

BUILD="$ROOT/$TAG/per_task"
mkdir -p "$BUILD"
NTASK=$(echo $TASKS | wc -w); NB=$(echo $BUCKETS | wc -w)
echo "=== $TAG | $NTASK tasks x $NB buckets = $((NTASK*NB)) builds | -P $NPROC | ${TOK_PER_BUCKET} tok/bucket ==="

bucket_tokens() {
  case "$1" in 2k) echo 2048;; 4k) echo 4096;; 8k) echo 8192;; 16k) echo 16384;;
               32k) echo 32768;; 64k) echo 65536;; 128k) echo 131072;; 256k) echo 262144;;
               *) echo 0;; esac
}

JOBS="$ROOT/$TAG/.jobs"
: > "$JOBS"
for task in $TASKS; do
  for b in $BUCKETS; do
    TOK=$(bucket_tokens "$b")
    [ "$TOK" = 0 ] && { echo "unknown bucket $b"; exit 2; }
    echo "$task $b $(( TOK_PER_BUCKET / TOK ))" >> "$JOBS"
  done
done

export PY BUILD
run_one() {
  task="$1"; b="$2"; n="$3"
  log="$BUILD/_logs/${task}_${b}.log"; mkdir -p "$BUILD/_logs"
  if [ -s "$BUILD/_b$b/$task/train.jsonl" ]; then echo "  [skip] $task@$b already built"; return 0; fi
  $PY -m ctc.data.cli build --task "$task" --split train --rungs "$b" --train "$n" \
     --pool auto --out "$BUILD/_b$b" > "$log" 2>&1 \
    && echo "  [ok]   $task@$b  n=$n" \
    || echo "  [FAIL] $task@$b  n=$n  -> $log"
}
export -f run_one

# `--pool auto` downloads a seed pool per task; letting 65 processes race on a cold HF cache would
# fetch the same pools many times over. One serial warm-up pass per DISTINCT task at the cheapest
# bucket fills the cache, then everything else runs wide against it.
echo "=== warming the seed-pool cache (one bucket per task, serial) ==="
FIRST_B=$(echo $BUCKETS | awk '{print $1}')
for task in $TASKS; do
  grep -E "^$task $FIRST_B " "$JOBS" | while read -r t b n; do run_one "$t" "$b" "$n"; done
done

echo "=== building the rest, $NPROC-way parallel ==="
xargs -a "$JOBS" -n3 -P "$NPROC" bash -c 'run_one "$0" "$1" "$2"'

echo "=== merging buckets per task ==="
for task in $TASKS; do
  mkdir -p "$BUILD/$task"; : > "$BUILD/$task/train.jsonl"
  for b in $BUCKETS; do
    f="$BUILD/_b$b/$task/train.jsonl"
    [ -s "$f" ] && cat "$f" >> "$BUILD/$task/train.jsonl"
  done
  echo "  $task: $(wc -l < "$BUILD/$task/train.jsonl") rows over $NB buckets"
done

echo "=== assembling the mix ==="
$PY "$REPO/src/scripts/data/hybridish/build_ctc_sft_mix.py" \
  --root "$BUILD" --tasks $TASKS --allow-spec-collision \
  --band "$(echo $BUCKETS | tr ' ' '-')" --out "$ROOT/$TAG/mix.jsonl"

echo "=== next: tokenise (query_position=both applies HERE) ==="
echo "  $PY $REPO/src/scripts/data/hybridish/convert_ctc_to_sft_completion.py \\"
echo "      --mix $ROOT/$TAG/mix.jsonl --query-position both --verify --out $ROOT/$TAG/shards"
echo "=== DONE $TAG -> $ROOT/$TAG/mix.jsonl $(date -u '+%F %T')Z ==="
