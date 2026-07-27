#!/bin/bash
# Tokenize the hotpotqa RETRIEVAL training pool into Llama document-chunked dense SFT shards.
# hotpotqa routes through --task retrieval (metric gold_id_f1); this is the O(N) half of the CTC
# hypothesis, the counterpart to the O(N*M) qdmatch row.
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=cubbins
#SBATCH --gres=gpu:H200:0
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --job-name=llama-tok-hpqa
#SBATCH --output=/data/prasann/joblogs/llama_tok_hpqa_%j.log
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
LOGDIR=/scratch/users/prasann/ctc_llama_logs
mkdir -p "$LOGDIR"
SRC=/data/prasann/joblogs/llama_tok_hpqa_${SLURM_JOB_ID}.log
( while true; do cp -f "$SRC" "$LOGDIR/tok_hpqa.log" 2>/dev/null; sleep 20; done ) &
trap 'cp -f "$SRC" "$LOGDIR/tok_hpqa.log" 2>/dev/null; kill %1 2>/dev/null' EXIT
ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$ENV" ] || ENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
export PATH="$ENV/bin:$PATH" PYTHONPATH="$REPO/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HOME=/data/prasann
TOK=/scratch/users/prasann/hf_models/Llama-3.2-3B-marker-tok
SEQ_LEN=${SEQ_LEN:-24576}
RAW=${RAW:-$REPO/debug/ctc_multifamily/hpqa_raw}
OUTDIR=/data/prasann/ctc_llama/shards/hotpotqa_train_llama
echo "=== HOST=$(hostname) SEQ_LEN=$SEQ_LEN START=$(date '+%F %T') ==="
ls -la "$RAW"
N=$(cat "$RAW"/*.jsonl 2>/dev/null | wc -l); echo "raw lines: $N"
[ "$N" -gt 0 ] || { echo "FATAL: no raw hotpotqa jsonl in $RAW"; exit 2; }
rm -rf "$OUTDIR"; mkdir -p "$OUTDIR"
cd "$REPO"
python src/scripts/data/convert_unified_to_document_landmark.py \
  --emit dense --task retrieval --chunk-by document --cot-mode none \
  --marker-set llama --tokenizer "$TOK" \
  --seq-len "$SEQ_LEN" --num-proc 8 --shard-tokens 60000000 \
  --input-jsonl "$RAW"/*.jsonl --out-dir "$OUTDIR"
echo "convert rc=$?"
cat "$OUTDIR/metadata.json"; cp -f "$OUTDIR/metadata.json" "$LOGDIR/shard_meta_hotpotqa.json"
DST=/scratch/users/prasann/ctc_llama/shards/hotpotqa_train_llama
mkdir -p "$DST"; cp -f "$OUTDIR"/* "$DST"/ && echo "staged -> $DST"
touch "$LOGDIR/HPQA_SHARD_READY"
echo "=== DONE $(date '+%F %T') ==="
