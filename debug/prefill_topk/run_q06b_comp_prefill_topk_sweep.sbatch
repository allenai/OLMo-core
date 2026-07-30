#!/bin/bash
# Prefill-wide top-k sweep for the local COMPRESSIVE-landmark Qwen3-0.6B contradiction-n20 SFT.
#
# Question: the shipped landmark eval applies hard top-k block retrieval only at DECODE; the prefill
# still soft-gates every prompt token over all past blocks. How much accuracy do we lose when top-k
# is applied to the WHOLE prefill (the honestly sparse setting)?
#
# Each config re-runs the same 488-example contradiction-n20 eval, only changing how attention is
# restricted. Results -> debug/prefill_topk/results/<tag>.json.
#
#   sbatch debug/prefill_topk/run_q06b_comp_prefill_topk_sweep.sbatch
#SBATCH --job-name=q06b-prefill-topk
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=mooney
#SBATCH --gres=gpu:H200:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/data/prasann/joblogs/q06b_prefill_topk_%j.log
set -uo pipefail

REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
CR=/scratch/users/prasann/corpus-reasoning
MODEL_PATH="${MODEL_PATH:-/data/prasann/olmo_ckpts/q06b-comp-contra-n20-sft-local/step750}"
CONTRA_DATA="${CONTRA_DATA:-data/contradiction_eval_pubmed_both_n20_k3.jsonl}"
NSAMPLES="${NSAMPLES:-488}"      # the WHOLE held-out set (eval_size=488), not the default 100
NGPU="${NGPU:-4}"
OUTDIR="$REPO/debug/prefill_topk/results"
mkdir -p "$OUTDIR"

OLMOENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$OLMOENV" ] || OLMOENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
export PATH="$OLMOENV/bin:$PATH"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
# $REPO/src -> olmo_core (this tree, with landmark_prefill_topk); $REPO/src/scripts -> ctc_eval
export PYTHONPATH="$REPO/src:$REPO/src/scripts:$CR"
export TRITON_CACHE_DIR=/data/prasann/triton_cache
export HOME=/data/prasann          # keep JIT/HF caches off NFS (see local_cluster.md)

mapfile -t CLEAN < <(nvidia-smi --query-gpu=memory.used,uuid --format=csv,noheader,nounits | awk -F', ' '$1+0 < 2000 {print $2}')
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${CLEAN[*]:0:$NGPU}")
echo "HOST=$(hostname) MODEL=$MODEL_PATH GPUS=$CUDA_VISIBLE_DEVICES START=$(date '+%F %T')"

cd "$CR"

# tag | extra flags
#   prefill top-k is OFF unless --prefill-topk* is passed; --decode-topk-fraction 0 = dense decode.
CONFIGS=(
  "dense_everywhere|--decode-topk-fraction 0"
  "baseline_decode_topk10pct|"
  "prefill_topk10pct_alpha0.1|--prefill-topk-fraction 0.1"
  "prefill_topk10pct_alpha0|--prefill-topk-fraction 0.1 --prefill-nonselected-mass 0"
  "prefill_topk50pct_alpha0|--prefill-topk-fraction 0.5 --prefill-nonselected-mass 0"
  "prefill_topk25pct_alpha0|--prefill-topk-fraction 0.25 --prefill-nonselected-mass 0"
  "prefill_topk25pct_alpha0.1|--prefill-topk-fraction 0.25"
  "prefill_topk4_alpha0|--prefill-topk 4 --prefill-nonselected-mass 0"
  "prefill_topk2_alpha0|--prefill-topk 2 --prefill-nonselected-mass 0"
  "prefill_topk1_alpha0|--prefill-topk 1 --prefill-nonselected-mass 0"
)

for entry in "${CONFIGS[@]}"; do
  TAG="${entry%%|*}"
  EXTRA="${entry#*|}"
  OUT="$OUTDIR/q06b_comp_n20_${TAG}.json"
  if [ -f "$OUT" ]; then echo "SKIP $TAG (exists)"; continue; fi
  echo "=== [$TAG] $EXTRA  $(date '+%T') ==="
  torchrun --nproc_per_node="$NGPU" --master_port=$((29500 + RANDOM % 1000)) \
    "$REPO/debug/prefill_topk/eval_lc_native_prefill_topk.py" \
    --model-path "$MODEL_PATH" \
    --out "$OUT" \
    --tokenizer Qwen/Qwen3-0.6B \
    --max-length 4096 \
    --max-test-samples "$NSAMPLES" \
    --batch-size 1 \
    --skip-ruler --skip-gen \
    --contra-data "$CONTRA_DATA" \
    --nq-data /nonexistent_skip_nq \
    $EXTRA
  echo "=== [$TAG] rc=$? $(date '+%T') ==="
done

echo "DONE $(date '+%F %T')"
