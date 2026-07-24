#!/bin/bash
# Prepare the plain Qwen3-4B distcp base for the contradiction comparison:
#   1) convert HF Qwen/Qwen3-4B-Base -> olmo-core distcp (with HF-parity validation on GPU)
#   2) trained-marker repair (box_start/box_end rows are untrained in stock Qwen3; the dense
#      box-marker SFT format needs them distinguishable AND in-distribution in norm, else even
#      plain-causal training flatlines -- see records/marker-embedding-norm-bug).
# Single process, 1 GPU, berkeleynlp. Writes the final base to /scratch (read in place at train).
#SBATCH --job-name=q3-4b-baseprep
#SBATCH --partition=berkeleynlp
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/users/prasann/ctc_qwen_compare/logs/base_prep_qwen3_4b_%j.log
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$ENV" ] || ENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$REPO/src"
export HF_HOME=/scratch/users/prasann/huggingface-cache
unset TRANSFORMERS_CACHE   # stale profile value points elsewhere -> repo-id offline lookups miss
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # rely on the pre-downloaded cache
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR=/data/prasann/triton_cache
mkdir -p "$TRITON_CACHE_DIR" 2>/dev/null || true

BASES=/scratch/users/prasann/ctc_qwen_compare/bases
RAW=$BASES/qwen3-4b-base-raw
FIXED=$BASES/qwen3-4b-base-trainedmark
mkdir -p "$BASES" /scratch/users/prasann/ctc_qwen_compare/logs
cd "$REPO"

# Local flat model dir (symlink to the cached snapshot) -- offline-safe, avoids the hub-cache
# repo-id offline-resolver bug on air-gapped compute nodes.
HFDIR=/scratch/users/prasann/hf_models/Qwen3-4B-Base
echo "=== [1/3] convert $HFDIR -> $RAW  (validate on GPU) $(date) ==="
python src/corpus_reasoning/train/convert_hf_to_olmo.py \
  --base-model "$HFDIR" --out "$RAW" || { echo "CONVERT FAILED"; exit 2; }

# Locate the distcp dir (the one holding .metadata): either $RAW or $RAW/model_and_optim.
if [ -f "$RAW/.metadata" ]; then BASE_DISTCP="$RAW";
elif [ -f "$RAW/model_and_optim/.metadata" ]; then BASE_DISTCP="$RAW/model_and_optim";
else echo "FATAL: no .metadata under $RAW"; ls -R "$RAW" | head; exit 3; fi
echo "converted distcp = $BASE_DISTCP"

echo "=== [2/3] trained-marker repair -> $FIXED/model_and_optim $(date) ==="
python src/scripts/data/fix_marker_embeddings.py \
  --base "$BASE_DISTCP" --out "$FIXED" --model-size 4B || { echo "MARKER FIX FAILED"; exit 4; }

echo "=== [3/3] verify $(date) ==="
if [ -f "$FIXED/model_and_optim/.metadata" ]; then
  echo "OK: $FIXED/model_and_optim/.metadata present"
  du -sh "$FIXED/model_and_optim"
else
  echo "FATAL: repaired base missing .metadata"; exit 5
fi
echo "=== DONE $(date). Final Qwen3-4B base for --base-checkpoint: $FIXED/model_and_optim ==="
