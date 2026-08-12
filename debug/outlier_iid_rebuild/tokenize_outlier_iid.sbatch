#!/bin/bash
# Tokenize the iid-rebuilt outlier training data into SFT shards.
#
# Drop-in replacement for single_task_ladders_v2/outlier, so it must match that shard's build
# exactly except for the source jsonl. From its metadata.json:
#     tokenizer Qwen/Qwen3-0.6B, eos 151643, median_len 16987, max_len 39522
# -> the same converter (corpus_reasoning/data/convert_unified_to_sft.py, identified by its
# metadata schema: input / median_len / max_len) at the 40960 SFT window, query_position both.
#
# The Qwen3 tokenizer is the RESULTS-HUB lineage. The CTC suite tokenizes the same tasks with the
# Qwen3.5 tokenizer + qwen3_5 marker set; that is a separate shard and is NOT produced here.
#SBATCH --job-name=outlier-iid-tok
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --nodelist=mooney
#SBATCH --output=/data/prasann/joblogs/outlier_iid_tok_%j.log
#SBATCH --chdir=/data/prasann
set -uo pipefail

export HOME=/data/prasann/home
export HF_HOME=/data/prasann/hf_cache
export TMPDIR=/data/prasann/tmp
mkdir -p "$HOME" "$TMPDIR" /data/prasann/joblogs

GENSRC=/data/prasann/repo/OLMo-core/src
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-eval/bin/python
SRC=/data/prasann/outlier_iid/outlier_iid_n14-220_k3_20000.jsonl
OUT=/data/prasann/outlier_iid/tokenized

echo "=== [outlier-iid-tok] HOST=$(hostname) START=$(date '+%F %T') ==="
ls -la "$SRC" || { echo "!!! rebuilt jsonl missing: $SRC"; exit 1; }
mkdir -p "$OUT"

PYTHONPATH="$GENSRC" $PY "$GENSRC/corpus_reasoning/data/convert_unified_to_sft.py" \
  --task outlier \
  --input "$SRC" \
  --out-dir "$OUT" \
  --tokenizer Qwen/Qwen3-0.6B \
  --max-seq-len 40960 \
  --query-position both || { echo "!!! tokenization FAILED"; exit 1; }

echo ""
echo "=== new shard metadata (compare to the original below) ==="
cat "$OUT/metadata.json"
echo ""
echo "ORIGINAL single_task_ladders_v2/outlier: num_instances 19981, num_tokens 340064902,"
echo "  num_loss_tokens 1092718, median_len 16987, max_len 39522, tokenizer Qwen/Qwen3-0.6B"
echo "A large shift in median_len/num_tokens is EXPECTED and fine (the corpus composition changed);"
echo "a shift in num_instances or a spike in num_skipped is NOT -- investigate before using."
ls -la "$OUT"
echo "=== [outlier-iid-tok] DONE $(date '+%F %T') ==="
