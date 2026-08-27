#!/bin/bash
# Compose the length-mix arms (jsonl prefixes+shuffle) and tokenize each into its own shard dir.
# Runs after the rung tokenization job (dependency set at submit time).
#SBATCH --job-name=outlier-lm-arms
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --nodelist=mooney
#SBATCH --output=/data/prasann/joblogs/outlier_lm_arms_%j.log
#SBATCH --chdir=/data/prasann
set -uo pipefail
export HOME=/data/prasann/home HF_HOME=/data/prasann/hf_cache TMPDIR=/data/prasann/tmp
GENSRC=/data/prasann/repo/OLMo-core/src
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-eval/bin/python
WORK=/data/prasann/outlier_lengthmix

echo "=== [arms] START $(date '+%F %T') ==="
$PY /data/prasann/repo/OLMo-core/compose_arms_lengthmix.py --work "$WORK" || exit 1

for f in "$WORK"/arms/*.jsonl; do
  ARM=$(basename "$f" .jsonl)
  OUT="$WORK/arms_tokenized/$ARM"
  [ -s "$OUT/metadata.json" ] && { echo "[skip] $ARM"; continue; }
  mkdir -p "$OUT"
  echo "--- tokenizing arm $ARM $(date '+%T') ---"
  PYTHONPATH="$GENSRC" $PY "$GENSRC/corpus_reasoning/data/convert_unified_to_sft.py" \
    --task outlier --input "$f" --out-dir "$OUT" \
    --tokenizer Qwen/Qwen3.5-0.8B-Base --max-seq-len 40960 --query-position after \
    --eos 248044 || { echo "!!! FAILED: $ARM"; exit 1; }
done
cp "$WORK/arms/MANIFEST.json" "$WORK/arms_tokenized/MANIFEST.json"
echo "=== [arms] DONE $(date '+%F %T') ==="
