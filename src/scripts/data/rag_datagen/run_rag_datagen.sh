#!/bin/bash
# Generate the NEAR RAG-QA training data (NQ + HotpotQA) as corpus-reasoning unified JSONL,
# via BM25 hard negatives over wikipedia-dpr-100w. Pure-CPU (pyserini BM25). Run on a Beaker
# CPU node by gantry with ragdatagen-env.yml (this dir) and weka mounted.
#
# The generators + their lib/ are vendored from PrasannS/corpus-reasoning (which Beaker cannot
# clone) so this runs entirely from OLMo-core. cd into this dir so PYTHONPATH picks up lib/ and
# align_hn_doc_lengths.py.
set -ex
cd "$(dirname "$0")"
export PYTHONPATH="$PWD"
OUT=/weka/oe-training-default/ai2-llm/checkpoints/prasanns/rag_jsonl
mkdir -p "$OUT"

echo "=== RLHN token count (set convert --target-tokens to this so NEAR matches FAR) ==="
python -c "import json; m=json.load(open('/weka/oe-training-default/ai2-llm/checkpoints/amandab/rlhn_sft_qwen_63k/metadata.json')); print('RLHN_NUM_TOKENS =', m.get('num_tokens'))" || echo "(could not read RLHN metadata.json)"

echo "=== generate NQ (k100, hn10) ==="
python -u generate_nq_training_data.py \
  --num-train 15000 --num-eval 500 \
  --num-docs 100 --num-hard-negatives 10 \
  --output-dir "$OUT"

echo "=== generate HotpotQA (k100, bridge, hn10) ==="
python -u generate_hotpotqa_data.py \
  --split train --num-examples 15000 --num-eval 500 \
  --num-docs 100 --question-type bridge --num-hard-negatives 10 \
  --output-dir "$OUT"

echo "=== written files ==="
ls -la "$OUT"
echo "ALL DONE"
