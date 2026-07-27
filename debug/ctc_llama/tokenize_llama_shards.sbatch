#!/bin/bash
# Tokenize the CTC-suite contradiction / qdmatch_hpqa mixed-n pools into document-chunked dense
# SFT shards with the LLAMA marker set + patched Llama tokenizer.
#   sbatch --array=0-1 --nodelist=cubbins debug/ctc_llama/tokenize_llama_shards.sbatch
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=llama-tok
#SBATCH --output=/data/prasann/joblogs/llama_tok_%a_%A.log
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
LOGDIR=/scratch/users/prasann/ctc_llama_logs
mkdir -p "$LOGDIR" /data/prasann/joblogs
SRC=/data/prasann/joblogs/llama_tok_${SLURM_ARRAY_TASK_ID}_${SLURM_ARRAY_JOB_ID}.log
( while true; do cp -f "$SRC" "$LOGDIR/tok_${SLURM_ARRAY_TASK_ID}.log" 2>/dev/null; sleep 20; done ) &
trap 'cp -f "$SRC" "$LOGDIR/tok_${SLURM_ARRAY_TASK_ID}.log" 2>/dev/null; kill %1 2>/dev/null' EXIT

ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$ENV" ] || ENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
export PATH="$ENV/bin:$PATH" PYTHONPATH="$REPO/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HOME=/data/prasann

TOK=${TOK:-/scratch/users/prasann/hf_models/Llama-3.2-3B-marker-tok}
SEQ_LEN=${SEQ_LEN:-18432}
# Read LIMIT source lines; examples longer than SEQ_LEN are dropped by the converter, so LIMIT is
# set above the ~10k target to land near it. Actual kept count is reported in metadata.json.
LIMIT=${LIMIT:-13500}
DATA=/data/prasann/ctc_suite_data
OUTROOT=${OUTROOT:-/data/prasann/ctc_llama/shards}

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0) NAME=contradiction; TASKKEY=contradiction
     IN="$DATA/contradiction_pool/contradiction_train_pubmed_realistic_n50-950_k3.jsonl" ;;
  1) NAME=qdmatch_hpqa;  TASKKEY=qdmatch
     IN="$DATA/qdmatch_hpqa/final20k/qdmatch_hpqa_train_20k_mixedn.jsonl" ;;
  *) echo "bad array index"; exit 1 ;;
esac
[ -f "$IN" ] || { echo "MISSING INPUT $IN"; exit 2; }
OUTDIR="$OUTROOT/${NAME}_train_llama"
echo "=== HOST=$(hostname) NAME=$NAME IN=$IN SEQ_LEN=$SEQ_LEN LIMIT=$LIMIT START=$(date '+%F %T') ==="
rm -rf "$OUTDIR"; mkdir -p "$OUTDIR"
cd "$REPO"
python src/scripts/data/convert_unified_to_document_landmark.py \
  --emit dense --task "$TASKKEY" --chunk-by document --cot-mode none \
  --marker-set llama --tokenizer "$TOK" \
  --seq-len "$SEQ_LEN" --limit "$LIMIT" --shard-tokens 60000000 \
  --input-jsonl "$IN" --out-dir "$OUTDIR"
rc=$?
echo "convert rc=$rc"
cat "$OUTDIR/metadata.json"
cp -f "$OUTDIR/metadata.json" "$LOGDIR/shard_meta_${NAME}.json" 2>/dev/null
du -sh "$OUTDIR"
echo "=== DONE $(date '+%F %T') rc=$rc ==="
