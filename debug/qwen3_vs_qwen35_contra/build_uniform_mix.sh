#!/bin/bash
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
export HF_HOME=/scratch/users/prasann/huggingface-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONWARNINGS=ignore PYTHONPATH=$REPO/src
cd "$REPO"
OUT=/scratch/users/prasann/ctc_qwen_compare
SRC=$OUT/raw/uniform_mix_8k_256k_10k.jsonl
SEQ=262144
echo "=== [1/3] gen 10k uniform 8k-256k $(date) ==="
$PY debug/qwen3_vs_qwen35_contra/gen_uniform_mix.py \
  --src /scratch/users/prasann/corpus-reasoning/data/contradiction_train_pubmed_recomb_n50_k3.jsonl \
  --out "$SRC" --num 10000 --n-min 175 --n-max 5400 --pool-abstracts 80000 --seed 42 || exit 2
echo "=== [2/3] tokenize qwen3 (seq_len $SEQ) $(date) ==="
$PY src/scripts/data/convert_unified_to_document_landmark.py \
  --emit dense --task contradiction --chunk-by document --cot-mode none --query-position both \
  --seq-len $SEQ --tokenizer Qwen/Qwen3-4B --marker-set qwen3 \
  --input-jsonl "$SRC" --out-dir "$OUT/contra_mix_qwen3_10k" || exit 3
echo "=== [3/3] tokenize qwen3.5 (seq_len $SEQ) $(date) ==="
$PY src/scripts/data/convert_unified_to_document_landmark.py \
  --emit dense --task contradiction --chunk-by document --cot-mode none --query-position both \
  --seq-len $SEQ --tokenizer /scratch/users/prasann/hf_models/Qwen3.5-4B-Base --marker-set qwen3_5 \
  --input-jsonl "$SRC" --out-dir "$OUT/contra_mix_qwen35_10k" || exit 3
echo "=== DONE $(date) ==="
for d in contra_mix_qwen3_10k contra_mix_qwen35_10k; do
  echo "--- $d ---"; grep -E "num_instances|num_dropped|max_example_len|min_example_len|marker_set" "$OUT/$d/metadata.json"
done
