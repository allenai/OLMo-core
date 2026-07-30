#!/bin/bash
# Build 2k-context contradiction SFT shards for the Qwen3-4B vs Qwen3.5-4B comparison.
# One raw source JSONL (5000 gold-pair examples resized to n=77 ~= 2k tokens), tokenized TWICE:
# once with the Qwen3 tokenizer (marker-set qwen3) and once with Qwen3.5 (marker-set qwen3_5).
# CPU-only; writes to /scratch (reachable from every node for training staging).
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
export HF_HOME=/scratch/users/prasann/huggingface-cache
export HF_DATASETS_CACHE=/scratch/users/prasann/huggingface-cache/datasets
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=$REPO/src
cd "$REPO"

OUT=/scratch/users/prasann/ctc_qwen_compare
RAW=$OUT/raw
mkdir -p "$RAW"

SRC=/scratch/users/prasann/corpus-reasoning/data/contradiction_train_pubmed_recomb_n50_k3.jsonl
SUBSET=$RAW/contra_recomb_subset5000.jsonl
NDOCS=77          # n=77 -> ~= 289 + 22.8*77 ~= 2045 tokens
NTRAIN=5000

echo "=== [1/4] subset $NTRAIN examples from recomb_n50 ==="
head -n $NTRAIN "$SRC" > "$SUBSET"
wc -l "$SUBSET"

echo "=== [2/4] expand n=50 -> n=$NDOCS (fresh PubMed fillers; downloads PubMedQA once) ==="
$PY src/corpus_reasoning/data/generate_pubmed_contradiction_data.py \
  --expand-from-train "$SUBSET" \
  --num-docs $NDOCS --num-contradictions 3 --mode both \
  --pool-abstracts 20000 --seed 42 \
  --output-dir "$RAW" || { echo "EXPAND FAILED"; exit 2; }
# generator writes contradiction_train_pubmed_both_n77_k3.jsonl into $RAW
SRC_JSONL=$RAW/contradiction_train_pubmed_both_n${NDOCS}_k3.jsonl
ls -la "$SRC_JSONL" || { echo "expanded jsonl missing"; exit 2; }

echo "=== [3/4] tokenize -> Qwen3 (marker-set qwen3) ==="
$PY src/scripts/data/convert_unified_to_document_landmark.py \
  --emit dense --task contradiction --chunk-by document --cot-mode none \
  --query-position both --seq-len 2816 \
  --tokenizer Qwen/Qwen3-4B --marker-set qwen3 \
  --input-jsonl "$SRC_JSONL" \
  --out-dir "$OUT/contra_2k_qwen3_n77_5k" || { echo "QWEN3 TOKENIZE FAILED"; exit 3; }

echo "=== [4/4] tokenize -> Qwen3.5 (marker-set qwen3_5) ==="
$PY src/scripts/data/convert_unified_to_document_landmark.py \
  --emit dense --task contradiction --chunk-by document --cot-mode none \
  --query-position both --seq-len 2816 \
  --tokenizer /scratch/users/prasann/hf_models/Qwen3.5-4B-Base --marker-set qwen3_5 \
  --input-jsonl "$SRC_JSONL" \
  --out-dir "$OUT/contra_2k_qwen35_n77_5k" || { echo "QWEN3.5 TOKENIZE FAILED"; exit 3; }

echo "=== DONE. metadata: ==="
for d in contra_2k_qwen3_n77_5k contra_2k_qwen35_n77_5k; do
  echo "--- $d ---"; cat "$OUT/$d/metadata.json"
done
