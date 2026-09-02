#!/bin/bash
# Held-out contradiction eval for the nested-FFN-MoE arms.
#
# Every arm is scored with plain causal attention (--variant full) on the 488-example
# contradiction set, no-cot. MAXLEN must exceed the prompt or f1 silently reads 0.000.
#
# CRITICAL: FFN_MOE_START/DIVISORS must match the values the run TRAINED with. The router is a
# learned parameter in the checkpoint -- mismatched flags either fail the load (wrong rung count)
# or leave the router unused, which scores a dense model and looks like the method "worked".
# Read them off the run's config.json "ffn_moe" block, which the trainer writes for this reason.
#
#   sbatch -w sneetches -p jsteinhardt -q preemptive_high --gres=gpu:H200:2 \
#     --export=ALL,RUNS="q4b-ffnmoe-v1",FFN_MOE_START=4 eval_q4b_ffnmoe_contra.sbatch
#SBATCH --job-name=q4b-ffnmoe-eval
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/data/prasann/joblogs/q4b_ffnmoe_eval_%x_%j.log

set -uo pipefail
MIRROR_DIR=/scratch/users/prasann/ffnmoe_logs
mkdir -p "$MIRROR_DIR" 2>/dev/null || true
_SRC_LOG="/data/prasann/joblogs/q4b_ffnmoe_eval_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
_mirror_log() { cp -f "$_SRC_LOG" "$MIRROR_DIR/eval_${SLURM_JOB_ID}.log" 2>/dev/null || true; }
( while true; do _mirror_log; sleep 30; done ) & _MIRROR_PID=$!

REPO="${REPO:-/accounts/projects/berkeleynlp/prasann/projects/OLMo-core}"
RUNS="${RUNS:?set RUNS=\"run1 run2\"}"
CKPT_ROOT="${CKPT_ROOT:-/data/prasann/ffnmoe_exp/runs}"
OUT_DIR=/data/prasann/ffnmoe_exp/eval_results
mkdir -p "$OUT_DIR"
EVAL_JSONL="${EVAL_JSONL:-/scratch/users/prasann/corpus-reasoning/data/contradiction_eval_pubmed_both_n100_k3.jsonl}"
NGPU="${NGPU:-2}"
MAXLEN="${MAXLEN:-8192}"
MAXNEW="${MAXNEW:-256}"
BATCH_SIZE="${BATCH_SIZE:-8}"
OUT_TAG="${OUT_TAG:-}"   # suffix on the result files, e.g. -bs1 for an unpadded re-score
# -1 = score dense. MUST match training for a routed run.
FFN_MOE_START="${FFN_MOE_START:--1}"
DIVISORS="${DIVISORS:-1,4,16,64}"

_finish() { cp -f "$OUT_DIR"/ffnmoe-* "$MIRROR_DIR/" 2>/dev/null || true; _mirror_log; kill $_MIRROR_PID 2>/dev/null; }
trap _finish EXIT

ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$ENV" ] || ENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$REPO/src"
export HF_HOME=/data/prasann/hf_cache
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
mapfile -t CLEAN < <(nvidia-smi --query-gpu=memory.used,uuid --format=csv,noheader,nounits | awk -F', ' '$1+0 < 2000 {print $2}')
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${CLEAN[*]:0:$NGPU}")
echo "=== HOST=$(hostname) RUNS='$RUNS' FFN_MOE_START=$FFN_MOE_START START=$(date '+%F %T') ==="

cd "$REPO"
for RUN in $RUNS; do
  MODEL_PATH="$CKPT_ROOT/$RUN"
  OUT="$OUT_DIR/ffnmoe-${RUN}${OUT_TAG}-full_native.json"
  if [ ! -f "$MODEL_PATH/model_and_optim/.metadata" ]; then
    echo "=== SKIP $RUN: no checkpoint at $MODEL_PATH ==="; continue
  fi
  # Cross-check the requested routing against what the run recorded, so a mismatch is loud.
  RECORDED=$(python -c "
import json,sys
try:
    c = json.load(open('$MODEL_PATH/config.json')).get('ffn_moe')
    print('dense' if c is None else f\"{c['start_layer']}|{c['divisors']}\")
except Exception as e:
    print('unknown')
" 2>/dev/null)
  WANT=$([ "$FFN_MOE_START" -lt 0 ] && echo "dense" || echo "${FFN_MOE_START}|${DIVISORS}")
  if [ "$RECORDED" != "unknown" ] && [ "$RECORDED" != "$WANT" ]; then
    echo "=== SKIP $RUN: trained routing '$RECORDED' != requested '$WANT' ==="; continue
  fi
  echo "=== [$RUN] eval start (routing=$WANT) $(date '+%F %T') ==="
  MASTER_PORT=$((29500 + RANDOM % 1000))
  torchrun --nproc_per_node="$NGPU" --master_port="$MASTER_PORT" \
    src/corpus_reasoning/eval/eval_lc_native_docchunk_contra.py \
    --variant full --model-path "$MODEL_PATH" --out "$OUT" \
    --tokenizer Qwen/Qwen3-4B --contra-data "$EVAL_JSONL" \
    --cot-mode none --contra-max-new-tokens "$MAXNEW" --max-length "$MAXLEN" \
    --max-test-samples 488 --batch-size "$BATCH_SIZE" \
    --ffn-moe-start-layer "$FFN_MOE_START" --ffn-moe-divisors "$DIVISORS" \
    --per-example-out "$OUT_DIR/ffnmoe-${RUN}${OUT_TAG}-full_per_example.json"
  echo "=== [$RUN] rc=$? -> $OUT $(date '+%F %T') ==="
  cp -f "$OUT_DIR"/ffnmoe-${RUN}-* "$MIRROR_DIR/" 2>/dev/null || true
done
echo "=== DONE $(date '+%F %T') ==="
