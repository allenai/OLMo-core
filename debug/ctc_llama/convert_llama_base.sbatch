#!/bin/bash
# Convert HF Llama-3.2-3B -> olmo-core model-only distcp WITH the marker-embedding repair, on a
# node with enough RAM (~30GB at fp32). Writes to node-local /data (fast) and mirrors the tiny
# audit JSON to /scratch so it is readable off-node.
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --job-name=llama-base-conv
#SBATCH --output=/data/prasann/joblogs/llama_base_conv_%j.log
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
LOGDIR=/scratch/users/prasann/ctc_llama_logs
mkdir -p "$LOGDIR" /data/prasann/joblogs
SRC=/data/prasann/joblogs/llama_base_conv_${SLURM_JOB_ID}.log
( while true; do cp -f "$SRC" "$LOGDIR/base_conv_$(hostname).log" 2>/dev/null; sleep 20; done ) &
trap 'cp -f "$SRC" "$LOGDIR/base_conv_$(hostname).log" 2>/dev/null; kill %1 2>/dev/null' EXIT

ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$ENV" ] || ENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
export PATH="$ENV/bin:$PATH" PYTHONPATH="$REPO/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HOME=/data/prasann

HFDIR=${HFDIR:-/scratch/users/prasann/hf_models/Llama-3.2-3B}
TOKDIR=${TOKDIR:-/scratch/users/prasann/hf_models/Llama-3.2-3B-marker-tok}
OUT=${OUT:-/data/prasann/ctc_llama/bases/llama32-3b-base-fixmark}
echo "=== HOST=$(hostname) OUT=$OUT START=$(date '+%F %T') ==="
mkdir -p "$(dirname "$OUT")"
cd "$REPO"
python src/scripts/train/memexpress/ctc_suite/convert_llama_base.py \
  --base-dir "$HFDIR" --tokenizer "$TOKDIR" --out "$OUT"
rc=$?
echo "convert rc=$rc"
if [ $rc -eq 0 ]; then
  cp -f "$OUT/marker_audit.json" "$LOGDIR/marker_audit_$(hostname).json" 2>/dev/null
  ls -la "$OUT/model_and_optim" | head -5
  du -sh "$OUT"
fi
echo "=== DONE $(date '+%F %T') rc=$rc ==="
