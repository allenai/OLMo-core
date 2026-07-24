#!/bin/bash
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
export HF_HOME=/scratch/users/prasann/huggingface-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONWARNINGS=ignore PYTHONPATH=$REPO/src
cd "$REPO"
SRC=/scratch/users/prasann/ctc_qwen_compare/raw/contradiction_train_pubmed_both_n77_k3.jsonl
OUT=/scratch/users/prasann/ctc_qwen_compare
SEQ=5632
echo "=== qwen3 (marker qwen3) seq_len=$SEQ ==="
$PY src/scripts/data/convert_unified_to_document_landmark.py \
  --emit dense --task contradiction --chunk-by document --cot-mode none \
  --query-position both --seq-len $SEQ \
  --tokenizer Qwen/Qwen3-4B --marker-set qwen3 \
  --input-jsonl "$SRC" --out-dir "$OUT/contra_2k_qwen3_n77_5k"
echo "=== qwen3.5 (marker qwen3_5) seq_len=$SEQ ==="
$PY src/scripts/data/convert_unified_to_document_landmark.py \
  --emit dense --task contradiction --chunk-by document --cot-mode none \
  --query-position both --seq-len $SEQ \
  --tokenizer /scratch/users/prasann/hf_models/Qwen3.5-4B-Base --marker-set qwen3_5 \
  --input-jsonl "$SRC" --out-dir "$OUT/contra_2k_qwen35_n77_5k"
echo "=== metadata ==="
for d in contra_2k_qwen3_n77_5k contra_2k_qwen35_n77_5k; do
  echo "--- $d ---"; grep -E "num_instances|max_example_len|min_example_len|num_dropped|marker_set|eos_token" "$OUT/$d/metadata.json"
done
echo "RETOKENIZE DONE"
