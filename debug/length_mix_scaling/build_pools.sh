#!/bin/bash
# Phase 0 for the length-composition experiment: build a FIXED long pool and a LARGE short pool.
#
# Design requirement being satisfied here: arms add increasing amounts of SHORT data on top of an
# IDENTICAL long pool. The short arms therefore need far more short data than exists anywhere --
# the production contradiction_train shard's <=8k slice is only ~60M tokens, but the biggest arm
# wants ~144M. Without generating more, short-heavy arms would silently loop their pool ~3x while
# the long-only arm did one pass, and that epoch-count confound would look exactly like a
# short-data effect. Hence: generate enough short data that every arm runs WITHOUT repetition.
#
# Tokenizer/marker-set = qwen3_5 to match the ONLY trustworthy baseline we have
# (ctc-s5-contra-full-4b, qwen3.5-4b, vLLM: 2k .849 / 8k .690 / 32k .335).
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
export HF_HOME=/scratch/users/prasann/huggingface-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONWARNINGS=ignore PYTHONPATH=$REPO/src
cd "$REPO"
OUT=/scratch/users/prasann/ctc_length_mix
SRC=/scratch/users/prasann/corpus-reasoning/data/contradiction_train_pubmed_recomb_n50_k3.jsonl
TOKZ=/scratch/users/prasann/hf_models/Qwen3.5-4B-Base
mkdir -p "$OUT/raw"

# ~47 tok/doc (calibration from the 64k iso build; verified post-hoc via metadata below).
#   short 2k-8k   -> n 44..174
#   long  16k-32k -> n 349..697
echo "=== [1/4] gen LONG pool: 1500 ex, 16k-32k (n 349..697) $(date '+%F %T') ==="
$PY debug/qwen3_vs_qwen35_contra/gen_uniform_mix.py --src "$SRC" \
  --out "$OUT/raw/long_16k_32k.jsonl" --num 1500 --n-min 349 --n-max 697 \
  --pool-abstracts 80000 --seed 11 || exit 2

echo "=== [2/4] gen SHORT pool: 30000 ex, 2k-8k (n 44..174) $(date '+%F %T') ==="
$PY debug/qwen3_vs_qwen35_contra/gen_uniform_mix.py --src "$SRC" \
  --out "$OUT/raw/short_2k_8k.jsonl" --num 30000 --n-min 44 --n-max 174 \
  --pool-abstracts 80000 --seed 12 || exit 3

# seq_len 40960 = the production training context; every example (<=32k) fits, packing does the rest.
echo "=== [3/4] tokenize LONG (qwen3_5, seq_len 40960) $(date '+%F %T') ==="
$PY debug/qwen3_vs_qwen35_contra/parallel_tokenize.py \
  --input-jsonl "$OUT/raw/long_16k_32k.jsonl" --out-dir "$OUT/pool_long" \
  --tokenizer "$TOKZ" --marker-set qwen3_5 --seq-len 40960 --nproc 7 || exit 4

echo "=== [4/4] tokenize SHORT (qwen3_5, seq_len 40960) $(date '+%F %T') ==="
$PY debug/qwen3_vs_qwen35_contra/parallel_tokenize.py \
  --input-jsonl "$OUT/raw/short_2k_8k.jsonl" --out-dir "$OUT/pool_short" \
  --tokenizer "$TOKZ" --marker-set qwen3_5 --seq-len 40960 --nproc 7 || exit 5

echo "=== POOL SUMMARY ==="
for p in pool_long pool_short; do
  $PY -c "
import json;m=json.load(open('$OUT/$p/metadata.json'))
ni,nt=m['num_instances'],m['num_tokens']
print(f'  $p: {ni} ex, {nt/1e6:.1f}M tok, {nt/ni:.0f} tok/ex, len {m[\"min_example_len\"]}..{m[\"max_example_len\"]}, dropped {m[\"num_dropped\"]}')"
done
echo "=== PHASE 0 DONE $(date '+%F %T') ==="
