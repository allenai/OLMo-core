#!/bin/bash
# Re-tokenize the FULL 20k contradiction pool (--limit 0). The n50-950 pool is much longer than the
# qdmatch one under the Llama tokenizer, so a 13.5k-line read kept only 5.8k examples at
# --seq-len 18432; reading all 20k lines lands near the ~10k target. Everything above 18432 tokens
# is dropped, which is the right slice anyway -- the eval ladder tops out at the 16k rung.
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=cubbins
#SBATCH --gres=gpu:H200:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --job-name=llama-tok-contra20k
#SBATCH --output=/data/prasann/joblogs/llama_tok_contra20k_%j.log
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
LOGDIR=/scratch/users/prasann/ctc_llama_logs
mkdir -p "$LOGDIR"
SRC=/data/prasann/joblogs/llama_tok_contra20k_${SLURM_JOB_ID}.log
( while true; do cp -f "$SRC" "$LOGDIR/tok_contra20k.log" 2>/dev/null; sleep 20; done ) &
trap 'cp -f "$SRC" "$LOGDIR/tok_contra20k.log" 2>/dev/null; kill %1 2>/dev/null' EXIT
ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$ENV" ] || ENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
export PATH="$ENV/bin:$PATH" PYTHONPATH="$REPO/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HOME=/data/prasann
TOK=/scratch/users/prasann/hf_models/Llama-3.2-3B-marker-tok
IN=/data/prasann/ctc_suite_data/contradiction_pool/contradiction_train_pubmed_realistic_n50-950_k3.jsonl
OUTDIR=/data/prasann/ctc_llama/shards/contradiction20k_train_llama
echo "=== HOST=$(hostname) START=$(date '+%F %T') ==="
rm -rf "$OUTDIR"; mkdir -p "$OUTDIR"
cd "$REPO"
python src/scripts/data/convert_unified_to_document_landmark.py \
  --emit dense --task contradiction --chunk-by document --cot-mode none \
  --marker-set llama --tokenizer "$TOK" \
  --seq-len 18432 --limit 0 --shard-tokens 60000000 \
  --input-jsonl "$IN" --out-dir "$OUTDIR"
echo "convert rc=$?"
cat "$OUTDIR/metadata.json"; cp -f "$OUTDIR/metadata.json" "$LOGDIR/shard_meta_contradiction20k.json"
# publish to shared /scratch for the training node
DST=/scratch/users/prasann/ctc_llama/shards/contradiction20k_train_llama
mkdir -p "$DST"; cp -f "$OUTDIR"/* "$DST"/ && echo "staged -> $DST"
touch "$LOGDIR/CONTRA20K_READY"
echo "=== DONE $(date '+%F %T') ==="
