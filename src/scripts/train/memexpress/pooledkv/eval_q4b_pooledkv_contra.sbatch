#!/bin/bash
# Full-attention held-out contradiction eval for the pooled-KV arms (the transfer test).
#
# Every arm is scored IDENTICALLY: plain causal attention (--variant full), the 488-example
# contradiction_eval_pubmed_both_n100_k3 set, no-cot, MAXLEN 8192 (> n100 prompt; a smaller value
# silently zeroes f1 -- see the goldgrad eval header). Checkpoints are read in place on the node
# that trained them (node-local /data).
#
#   sbatch --export=ALL eval_q4b_pooledkv_contra.sbatch
#   sbatch --export=ALL,RUNS="q4b-pkv-b1-e3" eval_q4b_pooledkv_contra.sbatch
#SBATCH --job-name=q4b-pooledkv-eval
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=mooney
#SBATCH --gres=gpu:H200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/data/prasann/joblogs/q4b_pooledkv_eval_%x_%j.log

set -uo pipefail
MIRROR_DIR=/scratch/users/prasann/pooledkv_logs
mkdir -p "$MIRROR_DIR" 2>/dev/null || true
_SRC_LOG="/data/prasann/joblogs/q4b_pooledkv_eval_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
_mirror_log() { cp -f "$_SRC_LOG" "$MIRROR_DIR/eval_${SLURM_JOB_ID}.log" 2>/dev/null || true; }
( while true; do _mirror_log; sleep 30; done ) & _MIRROR_PID=$!

REPO="${REPO:-/accounts/projects/berkeleynlp/prasann/projects/OLMo-core}"
RUNS="${RUNS:-q4b-pkv-full-e3 q4b-pkv-v1-e3 q4b-pkv-b1-e3}"
CKPT_ROOT="${CKPT_ROOT:-/data/prasann/pooledkv_exp/runs}"
OUT_DIR=/data/prasann/pooledkv_exp/eval_results
mkdir -p "$OUT_DIR"
EVAL_JSONL="${EVAL_JSONL:-/scratch/users/prasann/corpus-reasoning/data/contradiction_eval_pubmed_both_n100_k3.jsonl}"
NGPU="${NGPU:-2}"
MAXLEN="${MAXLEN:-8192}"
MAXNEW="${MAXNEW:-256}"
# Role-gated FFN arms MUST be scored with the same gate they trained with (-1 = off).
FFN_GATE="${FFN_GATE:--1}"
# Batched KV-cached decode (the driver's validated --batch-size path); bs=1 took 35-40min at 32k.
BATCH_SIZE="${BATCH_SIZE:-8}"

_finish() { cp -f "$OUT_DIR"/pooledkv-* "$MIRROR_DIR/" 2>/dev/null || true; _mirror_log; kill $_MIRROR_PID 2>/dev/null; }
trap _finish EXIT

ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$ENV" ] || ENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$REPO/src"
export HF_HOME=/data/prasann/hf_cache
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
mapfile -t CLEAN < <(nvidia-smi --query-gpu=memory.used,uuid --format=csv,noheader,nounits | awk -F', ' '$1+0 < 2000 {print $2}')
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${CLEAN[*]:0:$NGPU}")
echo "=== HOST=$(hostname) RUNS='$RUNS' CUDA=$CUDA_VISIBLE_DEVICES START=$(date '+%F %T') ==="

cd "$REPO"
for RUN in $RUNS; do
  MODEL_PATH="$CKPT_ROOT/$RUN"
  OUT="$OUT_DIR/pooledkv-${RUN}-full_native.json"
  if [ ! -f "$MODEL_PATH/model_and_optim/.metadata" ]; then
    echo "=== SKIP $RUN: no checkpoint at $MODEL_PATH ==="; continue
  fi
  echo "=== [$RUN] eval start $(date '+%F %T') ==="
  MASTER_PORT=$((29500 + RANDOM % 1000))
  torchrun --nproc_per_node="$NGPU" --master_port="$MASTER_PORT" \
    src/corpus_reasoning/eval/eval_lc_native_docchunk_contra.py \
    --variant full --model-path "$MODEL_PATH" --out "$OUT" \
    --tokenizer Qwen/Qwen3-4B --contra-data "$EVAL_JSONL" \
    --cot-mode none --contra-max-new-tokens "$MAXNEW" --max-length "$MAXLEN" \
    --max-test-samples 488 --batch-size "$BATCH_SIZE" \
    --ffn-gate-start-layer "$FFN_GATE" \
    --per-example-out "$OUT_DIR/pooledkv-${RUN}-full_per_example.json"
  echo "=== [$RUN] rc=$? -> $OUT $(date '+%F %T') ==="
  cp -f "$OUT_DIR"/pooledkv-${RUN}-* "$MIRROR_DIR/" 2>/dev/null || true
done
echo "=== DONE $(date '+%F %T') ==="
