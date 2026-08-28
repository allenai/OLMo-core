#!/bin/bash
# qdmatch_nq for the outlier-length-mix data-scaling framework: pools -> arms -> tokenized shards
# -> held-out eval rungs -> S3.  CPU only, no GPU needed.
#
# WHY A FULL REBUILD AND NOT A REUSE. Every shipped tokenized qdmatch_nq shard is the TRUNCATED
# one ([[qdmatch-nq-shard-truncated]]: 3 of 5 token_ids parts, all 5 mask parts, on /scratch +
# lambda + S3 alike) -- that, not underfitting, killed every prior qdmatch_nq run. Nothing
# tokenized is reused here; everything comes from the unified JSONL source.
#
# SOURCE = the p10 + CE-filtered NQ pipeline, NOT the banned 98%-hard files
# ([[nq-pipeline-10pct-hardneg-cefilter]]). BUILD_MATRIX.md's qdmatch command line still names
# `nq_train_k20_hn19_2500_aligned.jsonl` (95% hard) but the SHIPPED build did not use it: its own
# log says "19967 usable units", which is exactly nq_train_k25-202_clean.jsonl (19,967 examples,
# hard-ratio 0.097, CE-filtered). build_qdmatch_pools.py re-measures the ratio and hard-fails
# outside [0.05, 0.20], so a wrong source cannot slip through.
#
# Mirrors the outlier pipeline exactly for everything downstream of generation:
#   gen_rung_pools.sbatch -> build_qdmatch_pools.py   (fixed-N pools + 300-example heldout split)
#   compose_arms.py       -> compose_qdmatch_arms.py  (nested prefixes, SHUFFLE_SEED 7113)
#   compose_wave3.sbatch  -> the tokenize wave below  (convert_unified_to_sft.py, eos 248044,
#                                                      Qwen/Qwen3.5-0.8B-Base, max-seq-len 40960)
#
#   sbatch debug/outlier_lengthmix_scaling/build_qdmatch_lengthmix.sbatch
#SBATCH --job-name=qdnq-lengthmix
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=mooney
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=/data/prasann/joblogs/qdmatch_lengthmix_%j.log
#SBATCH --chdir=/data/prasann

set -uo pipefail

export HOME=/data/prasann/home
export TMPDIR=/data/prasann/tmp
export PYTHONWARNINGS=ignore
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HOME" "$TMPDIR" /data/prasann/joblogs

NFS_REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
GENSRC=/data/prasann/repo/OLMo-core/src
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-eval/bin/python   # same interpreter the
                                                                       # outlier arms were built
                                                                       # with; stdlib+transformers
AWS=/scratch/users/prasann/conda/envs/corpus-reasoning-train/bin/aws
WORK=/data/prasann/qdmatch_lengthmix
BIN=$WORK/bin
NQ_SRC=/scratch/users/prasann/nq_p10_20k
S3=s3://ai2-llm/checkpoints/prasanns/outlier_lengthmix
QWEN35_EOS=248044
TOKENIZER=Qwen/Qwen3.5-0.8B-Base

mkdir -p "$WORK" "$BIN" "$WORK/src" "$WORK/arms" "$WORK/arms_tokenized" "$WORK/tokenized"
echo "=== [qdnq-lm] HOST=$(hostname) JOB=${SLURM_JOB_ID:-none} START=$(date '+%F %T') ==="

# ---------- 0. stage code (small; the mooney repo clone is partial) ----------
for f in build_qdmatch_pools.py compose_qdmatch_arms.py verify_qdmatch_shard.py; do
  cp "$NFS_REPO/debug/outlier_lengthmix_scaling/$f" "$BIN/$f" || exit 1
done
[ -f "$GENSRC/corpus_reasoning/data/generate_qdmatch_data.py" ] || { echo "!!! no generator on node"; exit 1; }
grep -q 'add_argument("--eos"' "$GENSRC/corpus_reasoning/data/convert_unified_to_sft.py" \
  || { echo "!!! converter on mooney clone lacks --eos"; exit 1; }
export PYTHONPATH="$GENSRC"
export CTC_REPO_SRC="$GENSRC"

# ---------- 1. tokenizer preflight: node-local HF cache, fail fast ----------
mkdir -p /data/prasann/hf_local/transformers
rsync -a "/scratch/users/prasann/huggingface-cache/transformers/models--Qwen--Qwen3.5-0.8B-Base" \
  /data/prasann/hf_local/transformers/ 2>/dev/null
export TRANSFORMERS_CACHE=/data/prasann/hf_local/transformers
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if ! $PY -c "
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('$TOKENIZER'); print('tokenizer OK (node-local cache), vocab', len(t))
"; then
  echo "--- node-local tokenizer cache failed; falling back to the NFS cache"
  export TRANSFORMERS_CACHE=/scratch/users/prasann/huggingface-cache/transformers
  $PY -c "
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('$TOKENIZER'); print('tokenizer OK (NFS cache), vocab', len(t))
" || { echo "!!! tokenizer unavailable -- aborting before any work"; exit 1; }
fi

# ---------- 2. stage the p10 / CE-filtered NQ sources to node-local disk ----------
for f in nq_train_k25-202_clean.jsonl nq_validation_k25-202_600.jsonl; do
  if [ ! -s "$WORK/src/$f" ]; then
    echo "--- staging $f ($(date '+%T'))"
    cp "$NQ_SRC/$f" "$WORK/src/$f" || { echo "!!! stage failed: $f"; exit 1; }
  fi
  ls -la "$WORK/src/$f"
done

# ---------- 3. pools (q9 ~2k, q42 ~8k) + 300-example heldouts + held-out eval rungs ----------
if [ ! -s "$WORK/BUILD_REPORT.json" ]; then
  $PY -u "$BIN/build_qdmatch_pools.py" --work "$WORK" \
    --train-src "$WORK/src/nq_train_k25-202_clean.jsonl" \
    --eval-src  "$WORK/src/nq_validation_k25-202_600.jsonl" \
    --pool-2k 25300 --pool-8k 12300 --heldout 300 --eval-size 600 \
    || { echo "!!! pool build FAILED"; exit 1; }
else
  echo "[skip] pools already built (BUILD_REPORT.json present)"
fi
wc -l "$WORK"/qdmatch_nq_q*_train.jsonl "$WORK"/qdmatch_nq_q*_heldout.jsonl \
      "$WORK"/eval_rungs/qdmatch_nq/rung_*.jsonl

# ---------- 4. keep the shipped CTC-suite 500-example rungs alongside, for comparability ----------
SHIPPED=/data/prasann/ctc_suite_staged/eval_rungs/qdmatch_nq
if [ -d "$SHIPPED" ]; then
  mkdir -p "$WORK/eval_rungs/qdmatch_nq_ctcsuite500"
  for r in 2048 8192; do
    cp -u "$SHIPPED/rung_$r.jsonl" "$WORK/eval_rungs/qdmatch_nq_ctcsuite500/rung_$r.jsonl"
  done
  wc -l "$WORK"/eval_rungs/qdmatch_nq_ctcsuite500/*.jsonl
fi

# ---------- 5. arms ----------
$PY -u "$BIN/compose_qdmatch_arms.py" --work "$WORK" || { echo "!!! compose FAILED"; exit 1; }

# ---------- 6. tokenize (same converter/flags as the outlier arms) ----------
tok () {  # <out-subdir-under-arms_tokenized-or-tokenized> <input jsonl> <root>
  local NAME=$1 IN=$2 ROOT=$3
  local OUT="$WORK/$ROOT/$NAME"
  if [ -s "$OUT/metadata.json" ]; then echo "[skip] tok $NAME"; return 0; fi
  mkdir -p "$OUT"
  echo "--- tokenizing $NAME $(date '+%T')"
  $PY -u "$GENSRC/corpus_reasoning/data/convert_unified_to_sft.py" \
    --task qdmatch --input "$IN" --out-dir "$OUT" \
    --tokenizer "$TOKENIZER" --max-seq-len 40960 --eos $QWEN35_EOS \
    > "$WORK/logs_tok_$NAME.log" 2>&1 \
    || { echo "!!! tokenize FAILED: $NAME"; touch "$WORK/$ROOT/FAILED_$NAME"; return 1; }
  tail -1 "$WORK/logs_tok_$NAME.log"
}

# progress watcher: the per-arm converter logs are separate files so 8 parallel writers do not
# interleave, but the job log must still show progress (directive: log at items 1,2,5,10 + ETA).
watch_progress () {
  while true; do
    sleep 60
    printf '[watch %s]' "$(date '+%T')"
    for L in "$WORK"/logs_tok_*.log; do
      [ -f "$L" ] || continue
      N=$(basename "$L" .log); N=${N#logs_tok_}
      printf ' %s=%s' "$N" "$(tail -1 "$L" 2>/dev/null | tr -d '\n' | cut -c1-40)"
    done
    echo
  done
}
watch_progress & WATCH_PID=$!

# `wait` with explicit PIDs: a bare `wait` would also wait on the never-exiting watcher.
# wave A: the six requested arms + the two held-out CE shards
PIDS=()
tok q2k_1250 "$WORK/arms/q2k_1250.jsonl" arms_tokenized & PIDS+=($!)
tok q2k_2500 "$WORK/arms/q2k_2500.jsonl" arms_tokenized & PIDS+=($!)
tok q2k_5000 "$WORK/arms/q2k_5000.jsonl" arms_tokenized & PIDS+=($!)
tok q8k_1000 "$WORK/arms/q8k_1000.jsonl" arms_tokenized & PIDS+=($!)
tok q8k_2000 "$WORK/arms/q8k_2000.jsonl" arms_tokenized & PIDS+=($!)
tok q8k_4000 "$WORK/arms/q8k_4000.jsonl" arms_tokenized & PIDS+=($!)
tok q9_heldout  "$WORK/qdmatch_nq_q9_heldout.jsonl"  tokenized & PIDS+=($!)
tok q42_heldout "$WORK/qdmatch_nq_q42_heldout.jsonl" tokenized & PIDS+=($!)
wait "${PIDS[@]}"
echo "--- wave A done $(date '+%T')"

# wave B: headroom arms (ladder extension; failure here does not block wave A)
PIDS=()
tok q2k_10000 "$WORK/arms/q2k_10000.jsonl" arms_tokenized & PIDS+=($!)
tok q2k_20000 "$WORK/arms/q2k_20000.jsonl" arms_tokenized & PIDS+=($!)
tok q8k_8000  "$WORK/arms/q8k_8000.jsonl"  arms_tokenized & PIDS+=($!)
wait "${PIDS[@]}"
echo "--- wave B done $(date '+%T')"
kill $WATCH_PID 2>/dev/null

echo "=== TOKENIZE DONE LINES ==="
for d in "$WORK"/arms_tokenized/*/ "$WORK"/tokenized/*/; do
  [ -s "$d/metadata.json" ] && { printf '%s  ' "$(basename "$d")"; cat "$d/metadata.json" | tr -d '\n '; echo; }
done
ls "$WORK"/arms_tokenized/FAILED_* "$WORK"/tokenized/FAILED_* 2>/dev/null && echo "!!! SOME ARMS FAILED"

# ---------- 7. verify: no truncation, full context, answer+EOS ----------
echo "=== VERIFY ==="
VOK=1
for A in q2k_1250 q8k_4000; do
  echo "--- $A"
  $PY -u "$BIN/verify_qdmatch_shard.py" --shard "$WORK/arms_tokenized/$A" \
    --source "$WORK/arms/$A.jsonl" --tokenizer "$TOKENIZER" --index 0 || VOK=0
done
[ "$VOK" = 1 ] || { echo "!!! VERIFY FAILED -- not pushing to S3"; exit 1; }

# ---------- 8. publish ----------
cp "$WORK/arms/MANIFEST.json" "$WORK/arms_tokenized/MANIFEST_qdmatch_nq.json"
cp "$WORK/BUILD_REPORT.json" "$WORK/arms_tokenized/BUILD_REPORT_qdmatch_nq.json"
export AWS_PROFILE=S3
# NOTE: arms/ is SHARED with the outlier arms -- q-prefixed names, and the manifest is renamed,
# so nothing of the outlier build is overwritten. Never sync --delete here.
for A in "$WORK"/arms_tokenized/*/; do
  N=$(basename "$A"); case "$N" in q2k_*|q8k_*) ;; *) continue;; esac
  $AWS s3 sync "$A" "$S3/arms/$N" --only-show-errors || exit 1
done
$AWS s3 cp "$WORK/arms_tokenized/MANIFEST_qdmatch_nq.json" "$S3/arms/MANIFEST_qdmatch_nq.json" --only-show-errors
$AWS s3 cp "$WORK/arms_tokenized/BUILD_REPORT_qdmatch_nq.json" "$S3/arms/BUILD_REPORT_qdmatch_nq.json" --only-show-errors
$AWS s3 sync "$WORK/eval_rungs/qdmatch_nq" "$S3/eval_rungs/qdmatch_nq" --only-show-errors || exit 1
[ -d "$WORK/eval_rungs/qdmatch_nq_ctcsuite500" ] && \
  $AWS s3 sync "$WORK/eval_rungs/qdmatch_nq_ctcsuite500" "$S3/eval_rungs/qdmatch_nq_ctcsuite500" --only-show-errors
for N in q9 q42; do
  [ -s "$WORK/tokenized/${N}_heldout/metadata.json" ] && \
    $AWS s3 sync "$WORK/tokenized/${N}_heldout" "$S3/heldout/qdmatch_nq_${N}" --only-show-errors
done
echo "--- published ---"
$AWS s3 ls "$S3/arms/" | grep -i "q2k\|q8k\|qdmatch"
$AWS s3 ls --recursive "$S3/eval_rungs/qdmatch_nq"
echo "=== [qdnq-lm] DONE $(date '+%F %T') ==="
