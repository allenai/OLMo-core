#!/bin/bash
# Phase 0b: the Row C reference pool -- a UNIFORM 2k-32k mix, i.e. the production length
# composition. Generated from the SAME gold source + filler pool + tokenizer run as the long/short
# pools so Row C is apples-to-apples with rows A/B (importing the production shard instead would
# confound composition with data provenance).
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
export HF_HOME=/scratch/users/prasann/huggingface-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONWARNINGS=ignore PYTHONPATH=$REPO/src
cd "$REPO"
OUT=/scratch/users/prasann/ctc_length_mix
SRC=/scratch/users/prasann/corpus-reasoning/data/contradiction_train_pubmed_recomb_n50_k3.jsonl
TOKZ=/scratch/users/prasann/hf_models/Qwen3.5-4B-Base
echo "=== gen UNIFORM pool: 11000 ex, 2k-32k (n 44..697) $(date '+%F %T') ==="
$PY debug/qwen3_vs_qwen35_contra/gen_uniform_mix.py --src "$SRC" \
  --out "$OUT/raw/uniform_2k_32k.jsonl" --num 11000 --n-min 44 --n-max 697 \
  --pool-abstracts 80000 --seed 13 || exit 2
echo "=== tokenize UNIFORM (qwen3_5, seq_len 40960) $(date '+%F %T') ==="
$PY debug/qwen3_vs_qwen35_contra/parallel_tokenize.py \
  --input-jsonl "$OUT/raw/uniform_2k_32k.jsonl" --out-dir "$OUT/pool_uniform" \
  --tokenizer "$TOKZ" --marker-set qwen3_5 --seq-len 40960 --nproc 4 || exit 3
$PY -c "
import json;m=json.load(open('$OUT/pool_uniform/metadata.json'))
print('  pool_uniform: %d ex, %.1fM tok, %.0f tok/ex, len %d..%d'%(m['num_instances'],m['num_tokens']/1e6,m['num_tokens']/m['num_instances'],m['min_example_len'],m['max_example_len']))"
echo "=== PHASE 0b DONE $(date '+%F %T') ==="
