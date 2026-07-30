#!/bin/bash
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
export HF_HOME=/scratch/users/prasann/huggingface-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONWARNINGS=ignore PYTHONPATH=$REPO/src
cd "$REPO"
OUT=/scratch/users/prasann/ctc_qwen_compare
SRC=$OUT/raw/uniform_8k_64k_iso.jsonl
echo "=== gen 2000 uniform 8k-64k (n 175..1367) $(date) ==="
$PY debug/qwen3_vs_qwen35_contra/gen_uniform_mix.py \
  --src /scratch/users/prasann/corpus-reasoning/data/contradiction_train_pubmed_recomb_n50_k3.jsonl \
  --out "$SRC" --num 2000 --n-min 175 --n-max 1367 --pool-abstracts 30000 --seed 7 || exit 2
echo "=== tokenize qwen3 seq_len 65536 $(date) ==="
$PY src/scripts/data/convert_unified_to_document_landmark.py \
  --emit dense --task contradiction --chunk-by document --cot-mode none --query-position both \
  --seq-len 65536 --tokenizer Qwen/Qwen3-4B --marker-set qwen3 \
  --input-jsonl "$SRC" --out-dir "$OUT/contra_iso64k_qwen3_2k" || exit 3
grep -E "num_instances|num_dropped|max_example_len" "$OUT/contra_iso64k_qwen3_2k/metadata.json"
echo "=== 64K ISO DATA DONE $(date) ==="
