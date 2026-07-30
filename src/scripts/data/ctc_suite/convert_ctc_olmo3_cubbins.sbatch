#!/bin/bash
# CTC-suite OLMo3 arm: subsample the 20k contradiction / qdmatch_hpqa pools to ~10k examples each
# (capped at a 2k-16k rung length) and tokenize them with the OLMo/dolma2 marker set.
#
# The raw pools live ONLY on cubbins' node-local /data (they were never staged to /scratch), so this
# job is pinned to cubbins. Shards are written to /scratch so any node can train from them.
#
#   sbatch --partition=jsteinhardt --qos=preemptive --account=site -w cubbins \
#     src/scripts/data/ctc_suite/convert_ctc_olmo3_cubbins.sbatch
#SBATCH --job-name=ctc-olmo3-conv
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/data/prasann/joblogs/ctc_olmo3_conv_%j.log

set -uo pipefail
MIRROR=/scratch/users/prasann/ctc_suite_logs
mkdir -p "$MIRROR" 2>/dev/null || true
_SRC_LOG="/data/prasann/joblogs/ctc_olmo3_conv_${SLURM_JOB_ID}.log"
_DST_LOG="$MIRROR/ctc_olmo3_conv_${SLURM_JOB_ID}.log"
( while true; do cp -f "$_SRC_LOG" "$_DST_LOG" 2>/dev/null; sleep 20; done ) &
_MP=$!
trap 'cp -f "$_SRC_LOG" "$_DST_LOG" 2>/dev/null; kill $_MP 2>/dev/null' EXIT

REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$ENV" ] || ENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$REPO/src"
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HOME=/data/prasann/home_cache
mkdir -p "$HOME" 2>/dev/null || true

DATA=/data/prasann/ctc_suite_data
TOKENIZER=/scratch/users/prasann/hf_models/Olmo-3-1025-7B-docchunk
OUT=/scratch/users/prasann/ctc_suite_staged/shards_olmo3
TMP=/data/prasann/ctc_suite_olmo3_tmp
CONV=$REPO/src/scripts/data/convert_unified_to_document_landmark.py
SUB=$REPO/src/scripts/data/ctc_suite/subsample_pool.py
mkdir -p "$OUT" "$TMP"

TARGET=${TARGET:-10000}
MAXEST=${MAXEST:-16500}
SEQ_LEN=${SEQ_LEN:-20480}

echo "=== HOST=$(hostname) START=$(date '+%F %T') target=$TARGET max_est_tokens=$MAXEST ==="
echo "--- pools present? ---"
ls -la "$DATA/contradiction_pool/" 2>&1 | head -5
ls -la "$DATA/qdmatch_hpqa/final20k/" 2>&1 | head -5

run_one () {
  local NAME="$1" TASK="$2" SRC="$3"
  echo "############ $NAME ($TASK) ############"
  if [ ! -f "$SRC" ]; then echo "FATAL: missing pool $SRC"; return 2; fi
  python -u "$SUB" --input "$SRC" --output "$TMP/${NAME}_10k.jsonl" \
    --target "$TARGET" --max-est-tokens "$MAXEST" || return 3
  python -u "$CONV" --task "$TASK" \
    --input-jsonl "$TMP/${NAME}_10k.jsonl" \
    --out-dir "$OUT/${NAME}_train" \
    --marker-set olmo3 --tokenizer "$TOKENIZER" \
    --emit dense --chunk-by document --cot-mode none --seq-len "$SEQ_LEN" || return 4
  echo "--- metadata $NAME ---"
  cat "$OUT/${NAME}_train/metadata.json"
}

run_one contradiction contradiction \
  "$DATA/contradiction_pool/contradiction_train_pubmed_realistic_n50-950_k3.jsonl"
RC1=$?
run_one qdmatch_hpqa qdmatch \
  "$DATA/qdmatch_hpqa/final20k/qdmatch_hpqa_train_20k_mixedn.jsonl"
RC2=$?

echo "=== DONE contradiction_rc=$RC1 qdmatch_rc=$RC2 $(date '+%F %T') ==="
