#!/bin/bash
# oolong tokenization: --chunk-by line, and CRITICALLY no --item-regex (the converter default
# r"\|\|" is correct; a bare '||' matches every line and wraps the preamble as chunks -- the
# CHUNK_LEAK_AUDIT leak, fixed 2026-07-26).
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=cubbins
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --job-name=tok-oolong
#SBATCH --output=/data/prasann/xlong5/logs/tok_oolong_%A.log
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
OUT=/data/prasann/xlong5/${VARIANT_DIR:-shards}/oolong_train
mkdir -p "$OUT" /data/prasann/xlong5/logs
source /usr/local/linux/miniforge-3.13/etc/profile.d/conda.sh 2>/dev/null || source /system/linux/miniforge-3.13/etc/profile.d/conda.sh
conda activate /data/prasann/conda/envs/corpus-reasoning-olmo
export PYTHONPATH=$REPO/src HF_HOME=/data/prasann/hf_cache TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
FILES=$(find /data/prasann/xlong5/pools/oolong -name "*.jsonl" | sort)
echo "inputs=$(echo "$FILES" | wc -l) lines=$(cat $FILES | wc -l)"
rm -rf "$OUT"; mkdir -p "$OUT"
python $REPO/src/scripts/data/convert_unified_to_document_landmark.py \
  --emit dense --task oolong --chunk-by line --cot-mode none \
  --marker-set qwen3_5 --tokenizer /scratch/users/prasann/hf_models/Qwen3.5-4B-Base \
  --seq-len 262144 --shard-tokens 100000000 ${EXTRA_CONV:-} \
  --input-jsonl $FILES --out-dir "$OUT"
echo "convert rc=$? $(date '+%F %T')"
