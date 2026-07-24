#!/bin/bash
# Build 256k-context contradiction SFT shards for the Qwen3-4B vs Qwen3.5-4B comparison.
# 5000 gold-pair examples resized to n=5400 (~250k rendered tokens; seq_len 262144), tokenized
# once per tokenizer. CPU-only; writes to /scratch.
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
export HF_HOME=/scratch/users/prasann/huggingface-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONWARNINGS=ignore PYTHONPATH=$REPO/src
cd "$REPO"
OUT=/scratch/users/prasann/ctc_qwen_compare
RAW=$OUT/raw
SUBSET=$RAW/contra_recomb_subset5000.jsonl   # already built for the 2k run (same 5000 golds)
NDOCS=5400
SEQ=262144

echo "=== [1/3] expand n=50 -> n=$NDOCS (fresh PubMed fillers) $(date) ==="
$PY src/corpus_reasoning/data/generate_pubmed_contradiction_data.py \
  --expand-from-train "$SUBSET" --num-docs $NDOCS --num-contradictions 3 --mode both \
  --pool-abstracts 60000 --seed 42 --output-dir "$RAW" || { echo "EXPAND FAILED"; exit 2; }
SRC=$RAW/contradiction_train_pubmed_both_n${NDOCS}_k3.jsonl
ls -la "$SRC" || exit 2

echo "=== [2/3] tokenize -> Qwen3 (seq_len $SEQ) $(date) ==="
$PY src/scripts/data/convert_unified_to_document_landmark.py \
  --emit dense --task contradiction --chunk-by document --cot-mode none --query-position both \
  --seq-len $SEQ --tokenizer Qwen/Qwen3-4B --marker-set qwen3 \
  --input-jsonl "$SRC" --out-dir "$OUT/contra_256k_qwen3_n5400_5k" || { echo "Q3 TOK FAILED"; exit 3; }

echo "=== [3/3] tokenize -> Qwen3.5 (seq_len $SEQ) $(date) ==="
$PY src/scripts/data/convert_unified_to_document_landmark.py \
  --emit dense --task contradiction --chunk-by document --cot-mode none --query-position both \
  --seq-len $SEQ --tokenizer /scratch/users/prasann/hf_models/Qwen3.5-4B-Base --marker-set qwen3_5 \
  --input-jsonl "$SRC" --out-dir "$OUT/contra_256k_qwen35_n5400_5k" || { echo "Q35 TOK FAILED"; exit 3; }

echo "=== DONE $(date) ==="
for d in contra_256k_qwen3_n5400_5k contra_256k_qwen35_n5400_5k; do
  echo "--- $d ---"; grep -E "num_instances|num_dropped|max_example_len|marker_set" "$OUT/$d/metadata.json"
done
