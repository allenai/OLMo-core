#!/bin/bash
#SBATCH --job-name=build128k
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodes=1
#SBATCH --cpus-per-task=130
#SBATCH --mem=300G
#SBATCH --gres=gpu:0
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/users/prasann/ctc_qwen_compare/logs/build128k_%j.log
# 128k variant of the uniform mix (was 256k). Native context: Qwen3.5=262k (no ext),
# Qwen3-dense=32k (YaRN factor-4). n_max 2700 -> ~125k max tok (5% margin under 131072).
# gen + tokenize both tokenizers 128-way, all node-local /data, upload straight to S3.
set -uo pipefail
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
export HF_HOME=/scratch/users/prasann/huggingface-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONWARNINGS=ignore PYTHONPATH=$REPO/src
LOCAL=/data/prasann/ctc_qwen_compare_128k
mkdir -p "$LOCAL" /scratch/users/prasann/ctc_qwen_compare/logs || { echo "cannot mkdir on $(hostname)"; exit 1; }
SEQ=131072
SRC=$LOCAL/uniform_mix_8k_128k_10k.jsonl

echo "=== [1/4] gen 10k uniform 8k-128k (n_max 2700) $(date) ==="
$PY "$REPO/debug/qwen3_vs_qwen35_contra/gen_uniform_mix.py" \
  --src /scratch/users/prasann/corpus-reasoning/data/contradiction_train_pubmed_recomb_n50_k3.jsonl \
  --out "$SRC" --num 10000 --n-min 175 --n-max 2700 --pool-abstracts 80000 --seed 42 || exit 2

# stage qwen3.5 tokenizer to node-local (qwen3 tokenizer is a repo-id, cached under HF_HOME)
rm -rf "$LOCAL/Qwen3.5-4B-Base"; cp -r /scratch/users/prasann/hf_models/Qwen3.5-4B-Base "$LOCAL/Qwen3.5-4B-Base"

run_tok () {  # $1=out $2=tokenizer $3=marker
  local OUT=$LOCAL/$1
  rm -rf "$OUT" "$OUT.tmp"
  echo "=== tokenize $1 (seq_len $SEQ) $(date) ==="
  $PY "$REPO/debug/qwen3_vs_qwen35_contra/parallel_tokenize.py" \
    --input-jsonl "$SRC" --out-dir "$OUT" \
    --tokenizer "$2" --marker-set "$3" --seq-len $SEQ --nproc 128 || { echo "TOKENIZE $1 FAILED"; exit 3; }
  grep -E "num_instances|num_dropped|max_example_len" "$OUT/metadata.json"
  echo "=== S3 upload $1 $(date) ==="
  aws s3 sync "$OUT" "s3://ai2-llm/checkpoints/prasanns/ctc_suite/shards/$1" \
    && echo "S3 UPLOAD OK $1" || { echo "S3 UPLOAD $1 FAILED"; exit 4; }
  mkdir -p "/scratch/users/prasann/ctc_qwen_compare/$1"
  cp -f "$OUT/metadata.json" "/scratch/users/prasann/ctc_qwen_compare/$1/metadata.json"
}

echo "=== [2/4] tokenize qwen3 $(date) ==="
run_tok contra_mix_qwen3_10k_128k  Qwen/Qwen3-4B                  qwen3
echo "=== [3/4] tokenize qwen3.5 $(date) ==="
run_tok contra_mix_qwen35_10k_128k "$LOCAL/Qwen3.5-4B-Base"       qwen3_5

echo "=== [4/4] BUILD 128k DONE $(date) ==="
