#!/bin/bash
# Tokenize a banded 2k->256k pool into olmo-core dense document-chunked shards (Qwen3.5).
# ONE shard set serves BOTH the full/standard and chunked arms -- chunked-vs-standard is the
# train-time --variant flag, not a separate build (train_ctc_suite.py:18-21).
#   --use-titles OFF, --free-pad-repeat 0, --repeat-doc-text 1 (defaults): all three are known-bad
#   knobs if changed. NO --item-regex (see the oolong leak fix).
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=cubbins
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --job-name=xl5-tok
#SBATCH --output=/data/prasann/xlong5/logs/tok_%x_%A.log
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
: "${NAME:?}"; : "${TASK:?}"; : "${CHUNK:?}"
POOL=/data/prasann/xlong5/pools/$NAME
OUT=/data/prasann/xlong5/${VARIANT_DIR:-shards}/${NAME}_train
mkdir -p "$OUT" /data/prasann/xlong5/logs
source /usr/local/linux/miniforge-3.13/etc/profile.d/conda.sh 2>/dev/null || source /system/linux/miniforge-3.13/etc/profile.d/conda.sh
conda activate /data/prasann/conda/envs/corpus-reasoning-olmo
export PYTHONPATH=$REPO/src HF_HOME=/data/prasann/hf_cache TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
FILES=$(ls $POOL/*.jsonl 2>/dev/null)
[ -n "$FILES" ] || { echo "FATAL: no input jsonl under $POOL"; exit 2; }
echo "=== $NAME task=$TASK chunk=$CHUNK inputs=$(echo "$FILES" | wc -l) start=$(date '+%F %T') ==="
rm -rf "$OUT"; mkdir -p "$OUT"
python $REPO/src/scripts/data/convert_unified_to_document_landmark.py \
  --emit dense --task "$TASK" --chunk-by "$CHUNK" --cot-mode none \
  --marker-set qwen3_5 --tokenizer /scratch/users/prasann/hf_models/Qwen3.5-4B-Base \
  --seq-len 262144 --shard-tokens 100000000 ${EXTRA_CONV:-} \
  --input-jsonl $FILES --out-dir "$OUT"
echo "convert rc=$? $(date '+%F %T')"
ls -la "$OUT" | tail -20
