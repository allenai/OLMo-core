#!/bin/bash
# Does decode-time top-k retrieval explain the compressive-landmark collapse at long context?
#
# THE ASYMMETRY UNDER TEST. FastCompressiveLandmarkAttention is TRAINED with dense gating over every
# past block -- prefill never applies top-k (landmark_fast.py:1047-1050). At eval,
# GenerationConfig.landmark_top_k_fraction defaults to **0.1**, so every generated token can only
# reach the top 10% of blocks by landmark score. Nothing pins block 0, so the task instruction can be
# evicted too. That gap is pure eval-code behaviour: no model or data change, just the knob.
#
# The signature that prompted this: amandab's q35-4b-fastcomplm-xlong5-dolci25-256k scored
# contradiction f1=0.058 at the 128k rung (beaker 01KZQ4JTYJZB39GC7QH2T3GXR3) while emitting
# perfectly well-formed answers -- `[[131, 1660], [1014, 1946], [1463, 1562]]`, exactly k=3 pairs,
# ids in range. A model that knows the output format and cannot see the evidence.
#
# top_k_fraction=1.0 is the clean control: `_compressive_decode_probs` takes the top-k branch only
# when `n_lm > top_k`, and with top_k = ceil(1.0 * num_blocks) >= n_lm it falls through to dense
# gating with alpha=0 -- i.e. exactly the training-time math.
#
# Model: the LOCAL Qwen3-4B fast_compressive_landmark contra checkpoint (mem_freq=63). Qwen3, so the
# driver's Qwen3 landmark id (151860) is the correct one here -- this run does not exercise the
# Qwen3.5 id-mismatch bug, deliberately, so top-k is the only variable.
#
#   sbatch debug/complm_eval_debug/topk_ablation.sbatch
#SBATCH --partition=berkeleynlp
#SBATCH --nodelist=horton
#SBATCH --gres=gpu:H200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --job-name=complm-topk
#SBATCH --output=/data/prasann/complm_debug/logs/topk_%A.log
set -uo pipefail

REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
CKPT=/data/prasann/complm_debug/q4b-contra100-nocot-compressive/step549
CRROOT=/data/prasann/corpus-reasoning          # node-local copy of the eval data tree
OUT=/data/prasann/complm_debug/results
mkdir -p "$OUT" /data/prasann/complm_debug/logs

source /usr/local/linux/miniforge-3.13/etc/profile.d/conda.sh 2>/dev/null \
  || source /system/linux/miniforge-3.13/etc/profile.d/conda.sh
conda activate /data/prasann/conda/envs/corpus-reasoning-olmo
export PYTHONPATH=$REPO/src TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HOME=/data/prasann/hf_cache
export TMPDIR=/data/prasann/complm_debug/tmp; mkdir -p "$TMPDIR"

echo "=== HOST=$(hostname) GPUS=$CUDA_VISIBLE_DEVICES START=$(date '+%F %T') ==="
[ -d "$CKPT/model_and_optim" ] || { echo "FATAL: checkpoint not staged at $CKPT"; exit 2; }
# The staging copy is 47 GB / 128 distcp shards; a PARTIAL copy loads as a valid-looking checkpoint
# with missing tensors, so gate on the shard count rather than on the directory existing.
NSHARD=$(ls "$CKPT/model_and_optim"/*.distcp 2>/dev/null | wc -l)
[ "$NSHARD" -eq 128 ] || { echo "FATAL: $NSHARD/128 distcp shards staged -- copy incomplete"; exit 2; }
[ -f "$CRROOT/data/contradiction_eval_pubmed_both_n100_k3.jsonl" ] || {
  echo "FATAL: eval data missing under $CRROOT/data"; exit 3; }

# eval_size = 488 (the whole held-out contradiction file; accepted as-is per CLAUDE.md).
# Binomial SE at f1~0.5 on 488 is +-0.023, so only gaps well above ~0.05 are real.
for FRAC in 1.0 0.5 0.1; do
  echo
  echo "######## top_k_fraction=$FRAC  $(date '+%F %T') ########"
  torchrun --nproc_per_node=2 --master_port=$((29500 + RANDOM % 1000)) \
    $REPO/src/corpus_reasoning/eval/eval_lc_native_landmark.py \
    --model-path "$CKPT" \
    --out "$OUT/complm_contra_n100_topk${FRAC}.json" \
    --tokenizer Qwen/Qwen3-4B \
    --root "$CRROOT" \
    --contra-data data/contradiction_eval_pubmed_both_n100_k3.jsonl \
    --max-length 16384 \
    --max-test-samples 488 \
    --landmark-top-k-fraction "$FRAC" \
    --skip-ruler --skip-extra --skip-gen
  echo "rc=$? for frac=$FRAC $(date '+%F %T')"
done

echo
echo "=== SUMMARY ==="
python3 - "$OUT" <<'PY'
import glob, json, os, sys, math
root = sys.argv[1]
print(f"{'top_k_fraction':>16} {'contra f1':>10} {'eval_size':>10} {'SE':>7}")
for f in sorted(glob.glob(f"{root}/complm_contra_n100_topk*.json")):
    d = json.load(open(f))
    c = d.get("contradiction") or {}
    f1 = c.get("f1")
    n = (d.get("eval_size") or {}).get("contradiction")
    se = math.sqrt(f1 * (1 - f1) / n) if (f1 is not None and n) else None
    frac = os.path.basename(f).split("topk")[1].removesuffix(".json")
    print(f"{frac:>16} {f1 if f1 is None else round(f1,3):>10} {n:>10} "
          f"{'' if se is None else round(se,3):>7}")
PY
echo "DONE $(date '+%F %T')"
