#!/bin/bash
#SBATCH --job-name=tok128-qwen35
#SBATCH --partition=jsteinhardt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=130
#SBATCH --mem=300G
#SBATCH --gres=gpu:0
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/users/prasann/ctc_qwen_compare/logs/tok128_%j.log
set -uo pipefail
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
LOCAL=/data/prasann/ctc_qwen_compare
mkdir -p "$LOCAL" || { echo "cannot mkdir $LOCAL on $(hostname)"; exit 1; }

# Stage input + tokenizer to node-local /data so 128 workers don't hammer /scratch NFS.
echo "=== stage input + tokenizer to /data $(date) ==="
cp -f /scratch/users/prasann/ctc_qwen_compare/raw/uniform_mix_8k_256k_10k.jsonl "$LOCAL/"
rm -rf "$LOCAL/Qwen3.5-4B-Base"; cp -r /scratch/users/prasann/hf_models/Qwen3.5-4B-Base "$LOCAL/Qwen3.5-4B-Base"

OUT=$LOCAL/contra_mix_qwen35_10k
rm -rf "$OUT" "$OUT.tmp"
echo "=== tokenize 128-way -> $OUT (node-local) $(date) ==="
$PY "$REPO/debug/qwen3_vs_qwen35_contra/parallel_tokenize.py" \
  --input-jsonl "$LOCAL/uniform_mix_8k_256k_10k.jsonl" --out-dir "$OUT" \
  --tokenizer "$LOCAL/Qwen3.5-4B-Base" --marker-set qwen3_5 \
  --seq-len 262144 --nproc 128 || { echo "TOKENIZE FAILED"; exit 2; }
grep -E "num_instances|num_dropped|max_example_len" "$OUT/metadata.json"

# Upload straight to S3 from node-local (S3 is reachable from cubbins) -- skip /scratch entirely.
echo "=== S3 upload from /data $(date) ==="
aws s3 sync "$OUT" s3://ai2-llm/checkpoints/prasanns/ctc_suite/shards/contra_mix_qwen35_10k \
  && echo "S3 UPLOAD OK" || { echo "S3 UPLOAD FAILED"; exit 3; }
# also drop a small copy of just metadata.json to /scratch so the login node can confirm readiness
mkdir -p /scratch/users/prasann/ctc_qwen_compare/contra_mix_qwen35_10k
cp -f "$OUT/metadata.json" /scratch/users/prasann/ctc_qwen_compare/contra_mix_qwen35_10k/metadata.json
echo "=== tok128 DONE rc=$? $(date) ==="
