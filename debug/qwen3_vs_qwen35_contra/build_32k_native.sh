#!/bin/bash
# Build a NATIVE-context (32k) qwen3 contradiction mix, to test whether the ~1.07 plateau needs a
# non-native context at all. Mirrors build_64k_iso.sh exactly except seq_len 32768 and the n range,
# so the only difference vs the 64k iso data is the length distribution.
#   Qwen3-4B native max_position_embeddings = 32768 -> NO rope extension needed for this run.
#   Calibration from the 64k build: n=1367 <-> 64k  =>  ~46.8 tok/doc  =>  n=700 <-> 32k.
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
export HF_HOME=/scratch/users/prasann/huggingface-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONWARNINGS=ignore PYTHONPATH=$REPO/src
cd "$REPO"
OUT=/scratch/users/prasann/ctc_qwen_compare
SRC=$OUT/raw/uniform_8k_32k_native.jsonl
mkdir -p "$OUT/raw"

echo "=== gen 2000 uniform 8k-32k (n 175..700) $(date) ==="
$PY debug/qwen3_vs_qwen35_contra/gen_uniform_mix.py \
  --src /scratch/users/prasann/corpus-reasoning/data/contradiction_train_pubmed_recomb_n50_k3.jsonl \
  --out "$SRC" --num 2000 --n-min 175 --n-max 700 --pool-abstracts 30000 --seed 7 || exit 2

echo "=== tokenize qwen3 seq_len 32768 (parallel, nproc 7) $(date) ==="
$PY debug/qwen3_vs_qwen35_contra/parallel_tokenize.py \
  --input-jsonl "$SRC" --out-dir "$OUT/contra_native32k_qwen3_2k" \
  --tokenizer Qwen/Qwen3-4B --marker-set qwen3 --seq-len 32768 --nproc 7 || exit 3

grep -E "num_instances|num_dropped|max_example_len|min_example_len" "$OUT/contra_native32k_qwen3_2k/metadata.json"
echo "=== 32K NATIVE DATA DONE $(date) ==="
