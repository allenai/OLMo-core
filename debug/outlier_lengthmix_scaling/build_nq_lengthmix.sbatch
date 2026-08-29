#!/bin/bash
# Plain **nq** (retrieval-graded, p10 pipeline) for the outlier-length-mix data-scaling framework:
# pools -> arms -> tokenized shards -> held-out eval rungs -> S3.  CPU only, no GPU needed.
#
# Exact sibling of build_qdmatch_lengthmix.sbatch; the three task-specific differences are:
#   * build_prompt task is **retrieval**, NOT "nq" -- there is no "nq" task in build_prompt and it
#     would silently fall through to the generic QA instruction, producing a shard that trains the
#     model to emit free-text answers instead of `Relevant Document: [id]`.
#   * LENGTH comes from k = documents/example, produced by SHRINKING the 25-202-doc source rows
#     (build_v2_eval_ladders.py "shrink" mode). k is MEASURED, not assumed
#     ([[ctc-rung-labels-not-tokens]]): `build_nq_pools.py --calibrate` on 120 validation rows with
#     Qwen/Qwen3.5-0.8B-Base, query-position after ->
#         k=11 median 1752 | k=12 median 1907 | k=13 median 2063
#         k=42 median 6557 | k=48 median 7509 | k=50 median 7828 | k=52 median 8131
#     => k2k=12 (median 1907, 93% of the 2048 label), k8k=48 (median 7509, 92% of the 8192 label,
#        and the same k the shipped v2 nq ladder uses for its 8k rung).
#   * nq's unit is a QUERY and there are only 19,967 of them, so the 20k arm needs a ~1.02x reuse
#     factor (a few queries appear twice with different distractor subsets, appended at the END of
#     the pool so every smaller arm stays duplicate-free). The 300-example held-out split is carved
#     out BEFORE any reuse.
#
# SOURCE = the p10 + CE-filtered NQ pipeline ONLY ([[nq-pipeline-10pct-hardneg-cefilter]]);
# build_nq_pools.py re-measures the hard-negative ratio of both files and hard-fails outside
# [0.05, 0.20], so the retired 98%-hard NQ cannot slip through. Train/eval query disjointness is
# asserted (== 0 shared queries), not assumed.
#
#   sbatch debug/outlier_lengthmix_scaling/build_nq_lengthmix.sbatch
#SBATCH --job-name=nq-lengthmix
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=mooney
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --output=/data/prasann/joblogs/nq_lengthmix_%j.log
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
                                                                       # outlier + qdmatch arms
                                                                       # were built with
AWS=/scratch/users/prasann/conda/envs/corpus-reasoning-train/bin/aws
WORK=/data/prasann/nq_lengthmix
BIN=$WORK/bin
# the p10 sources are ALREADY node-local from the qdmatch build -- reuse them, do not re-copy
# 1.6 GB onto a 98%-full disk.
NQ_LOCAL=/data/prasann/qdmatch_lengthmix/src
NQ_NFS=/scratch/users/prasann/nq_p10_20k
S3=s3://ai2-llm/checkpoints/prasanns/outlier_lengthmix
QWEN35_EOS=248044
TOKENIZER=Qwen/Qwen3.5-0.8B-Base
K2K=${K2K:-12}
K8K=${K8K:-48}

mkdir -p "$WORK" "$BIN" "$WORK/arms" "$WORK/arms_tokenized" "$WORK/tokenized"
echo "=== [nq-lm] HOST=$(hostname) JOB=${SLURM_JOB_ID:-none} START=$(date '+%F %T') ==="
df -h /data | tail -1

# ---------- 0. stage code (small; the mooney repo clone is partial) ----------
for f in build_nq_pools.py compose_nq_arms.py verify_nq_shard.py; do
  cp "$NFS_REPO/debug/outlier_lengthmix_scaling/$f" "$BIN/$f" || exit 1
done
grep -q 'add_argument("--eos"' "$GENSRC/corpus_reasoning/data/convert_unified_to_sft.py" \
  || { echo "!!! converter on mooney clone lacks --eos"; exit 1; }
grep -q 'PASSAGE_TEMPLATE_ID' "$GENSRC/corpus_reasoning/lib/prompts.py" \
  || { echo "!!! prompts.py on mooney clone is stale"; exit 1; }
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

# ---------- 2. locate the p10 / CE-filtered NQ sources ----------
TRAIN_SRC=$NQ_LOCAL/nq_train_k25-202_clean.jsonl
EVAL_SRC=$NQ_LOCAL/nq_validation_k25-202_600.jsonl
for v in TRAIN_SRC EVAL_SRC; do
  P=${!v}
  if [ ! -s "$P" ]; then
    echo "--- $P missing; staging from NFS ($(date '+%T'))"
    mkdir -p "$WORK/src"
    cp "$NQ_NFS/$(basename "$P")" "$WORK/src/$(basename "$P")" || { echo "!!! stage failed"; exit 1; }
    eval "$v=$WORK/src/$(basename "$P")"
  fi
  ls -la "${!v}"
done

# ---------- 3. pools (k=$K2K ~2k, k=$K8K ~8k) + 300 heldout + held-out eval rungs ----------
if [ ! -s "$WORK/BUILD_REPORT.json" ]; then
  $PY -u "$BIN/build_nq_pools.py" --work "$WORK" \
    --train-src "$TRAIN_SRC" --eval-src "$EVAL_SRC" \
    --k2k "$K2K" --k8k "$K8K" \
    --pool-2k 20300 --pool-8k 8300 --heldout 300 --eval-size 600 --query-position after \
    || { echo "!!! pool build FAILED"; exit 1; }
  rm -rf "$WORK/tmp"          # the per-row variant spill files; ~1 GB, not an artifact
else
  echo "[skip] pools already built (BUILD_REPORT.json present)"
fi
wc -l "$WORK"/nq_n*_train.jsonl "$WORK"/nq_n*_heldout.jsonl "$WORK"/eval_rungs/nq/rung_*.jsonl
df -h /data | tail -1

# ---------- 4. arms ----------
$PY -u "$BIN/compose_nq_arms.py" --work "$WORK" || { echo "!!! compose FAILED"; exit 1; }

# ---------- 5. tokenize (same converter/flags as the outlier + qdmatch arms) ----------
tok () {  # <out-subdir> <input jsonl> <root under $WORK>
  local NAME=$1 IN=$2 ROOT=$3
  local OUT="$WORK/$ROOT/$NAME"
  if [ -s "$OUT/metadata.json" ]; then echo "[skip] tok $NAME"; return 0; fi
  mkdir -p "$OUT"
  echo "--- tokenizing $NAME $(date '+%T')"
  $PY -u "$GENSRC/corpus_reasoning/data/convert_unified_to_sft.py" \
    --task retrieval --input "$IN" --out-dir "$OUT" \
    --tokenizer "$TOKENIZER" --max-seq-len 40960 --query-position after --eos $QWEN35_EOS \
    > "$WORK/logs_tok_$NAME.log" 2>&1 \
    || { echo "!!! tokenize FAILED: $NAME"; touch "$WORK/$ROOT/FAILED_$NAME"; return 1; }
  tail -1 "$WORK/logs_tok_$NAME.log"
}

# progress watcher: per-arm converter logs are separate files so parallel writers do not
# interleave, but the job log must still show progress (directive: log early, with ETA).
watch_progress () {
  while true; do
    sleep 60
    printf '[watch %s]' "$(date '+%T')"
    for L in "$WORK"/logs_tok_*.log; do
      [ -f "$L" ] || continue
      N=$(basename "$L" .log); N=${N#logs_tok_}
      printf ' %s=%s' "$N" "$(tail -1 "$L" 2>/dev/null | tr -d '\n' | cut -c1-30)"
    done
    echo
  done
}
watch_progress & WATCH_PID=$!

# `wait` with explicit PIDs: a bare `wait` would also wait on the never-exiting watcher.
PIDS=()
tok nq2k_1250  "$WORK/arms/nq2k_1250.jsonl"  arms_tokenized & PIDS+=($!)
tok nq2k_2500  "$WORK/arms/nq2k_2500.jsonl"  arms_tokenized & PIDS+=($!)
tok nq2k_5000  "$WORK/arms/nq2k_5000.jsonl"  arms_tokenized & PIDS+=($!)
tok nq2k_10000 "$WORK/arms/nq2k_10000.jsonl" arms_tokenized & PIDS+=($!)
tok nq2k_20000 "$WORK/arms/nq2k_20000.jsonl" arms_tokenized & PIDS+=($!)
tok nq8k_1000  "$WORK/arms/nq8k_1000.jsonl"  arms_tokenized & PIDS+=($!)
tok nq8k_2000  "$WORK/arms/nq8k_2000.jsonl"  arms_tokenized & PIDS+=($!)
tok nq8k_4000  "$WORK/arms/nq8k_4000.jsonl"  arms_tokenized & PIDS+=($!)
tok nq8k_8000  "$WORK/arms/nq8k_8000.jsonl"  arms_tokenized & PIDS+=($!)
tok n2k_heldout "$WORK/nq_n2k_heldout.jsonl" tokenized & PIDS+=($!)
tok n8k_heldout "$WORK/nq_n8k_heldout.jsonl" tokenized & PIDS+=($!)
wait "${PIDS[@]}"
kill $WATCH_PID 2>/dev/null
echo "--- tokenize wave done $(date '+%T')"

echo "=== TOKENIZE DONE LINES ==="
for d in "$WORK"/arms_tokenized/*/ "$WORK"/tokenized/*/; do
  [ -s "$d/metadata.json" ] && { printf '%s  ' "$(basename "$d")"; tr -d '\n ' < "$d/metadata.json"; echo; }
done
ls "$WORK"/arms_tokenized/FAILED_* "$WORK"/tokenized/FAILED_* 2>/dev/null && echo "!!! SOME ARMS FAILED"
df -h /data | tail -1

# ---------- 6. verify: no truncation, full context, answer+EOS, answer-only loss ----------
echo "=== VERIFY ==="
VOK=1
for A in nq2k_1250 nq8k_4000; do
  echo "--- $A"
  $PY -u "$BIN/verify_nq_shard.py" --shard "$WORK/arms_tokenized/$A" \
    --source "$WORK/arms/$A.jsonl" --tokenizer "$TOKENIZER" --index 0 || VOK=0
done
[ "$VOK" = 1 ] || { echo "!!! VERIFY FAILED -- not pushing to S3"; exit 1; }

# ---------- 7. publish ----------
cp "$WORK/arms/MANIFEST.json"    "$WORK/arms_tokenized/MANIFEST_nq.json"
cp "$WORK/BUILD_REPORT.json"     "$WORK/arms_tokenized/BUILD_REPORT_nq.json"
export AWS_PROFILE=S3
# NOTE: arms/ is SHARED with the outlier (p*) and qdmatch (q*) arms -- nq-prefixed names and a
# renamed manifest, so nothing existing is overwritten. NEVER sync --delete here.
for A in "$WORK"/arms_tokenized/*/; do
  N=$(basename "$A"); case "$N" in nq2k_*|nq8k_*) ;; *) continue;; esac
  $AWS s3 sync "$A" "$S3/arms/$N" --only-show-errors || exit 1
done
$AWS s3 cp "$WORK/arms_tokenized/MANIFEST_nq.json"    "$S3/arms/MANIFEST_nq.json" --only-show-errors
$AWS s3 cp "$WORK/arms_tokenized/BUILD_REPORT_nq.json" "$S3/arms/BUILD_REPORT_nq.json" --only-show-errors
$AWS s3 sync "$WORK/eval_rungs/nq" "$S3/eval_rungs/nq" --only-show-errors || exit 1
for N in n2k n8k; do
  [ -s "$WORK/tokenized/${N}_heldout/metadata.json" ] && \
    $AWS s3 sync "$WORK/tokenized/${N}_heldout" "$S3/heldout/nq_${N}" --only-show-errors
done
echo "--- published ---"
$AWS s3 ls "$S3/arms/" | grep -i "nq2k\|nq8k\|_nq.json"
$AWS s3 ls --recursive "$S3/eval_rungs/nq"
echo "=== [nq-lm] DONE $(date '+%F %T') ==="
