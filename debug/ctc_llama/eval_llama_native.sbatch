#!/bin/bash
# Native (olmo_core) CTC eval for the LLAMA family. Same driver as
# debug/ctc_vllm_validation/eval_ctc_native.sbatch, but it passes the Llama marker ids and the
# PATCHED Llama tokenizer -- run_rung_eval.py defaults to the Qwen3.5 ids/tokenizer, and a
# wrong-tokenizer eval produces plausible numbers rather than a crash.
#
# Also: unique --master-port per job (concurrent evals otherwise collide on torchrun's default
# 29500 and all but one die in ~3s), and --batch-size 1 (the evaluator's left-pad id is the
# hardcoded Qwen PAD_TOKEN_ID, which is out of range for Llama's 128256-row vocab; at batch-size 1
# no pad token ever enters the sequence, so it is inert).
#
#   sbatch -w mooney --gres=gpu:H200:2 --export=ALL,CKPT=...,TASK=contradiction,VARIANT=dense,\
#     ARM=full,RUNG=2048,EVAL_JSONL=...,NGPU=2 debug/ctc_llama/eval_llama_native.sbatch
#SBATCH --job-name=llama-eval
#SBATCH --partition=jsteinhardt
#SBATCH --account=site
#SBATCH --qos=preemptive_high
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/data/prasann/joblogs/llama_eval_%x_%j.log
set -uo pipefail
MIRROR=/scratch/users/prasann/ctc_llama_logs
mkdir -p "$MIRROR" /data/prasann/joblogs
SRC=/data/prasann/joblogs/llama_eval_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log
DST=$MIRROR/eval_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log
( while true; do cp -f "$SRC" "$DST" 2>/dev/null || true; sleep 20; done ) &
trap 'cp -f "$SRC" "$DST" 2>/dev/null || true; kill %1 2>/dev/null' EXIT

REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
NGPU="${NGPU:-2}"
TASK="${TASK:?}"; VARIANT="${VARIANT:-dense}"; ARM="${ARM:-full}"; SCALE="${SCALE:-llama3.2-3b}"
RUNG="${RUNG:-2048}"; CKPT="${CKPT:?}"; EVAL_JSONL="${EVAL_JSONL:?}"
TOKENIZER="${TOKENIZER:-/scratch/users/prasann/hf_models/Llama-3.2-3B-marker-tok}"
DOC_START_ID="${DOC_START_ID:-128002}"; DOC_END_ID="${DOC_END_ID:-128003}"; EOS_ID="${EOS_ID:-128001}"
# Distinct per (task, arm, rung) so concurrent evals never overwrite each other's results.
OUTROOT="${OUTROOT:-$REPO/results/ctc_suite_llama}"
MAXLEN="${MAXLEN:-}"
MASTER_PORT="${MASTER_PORT:-$((29100 + RANDOM % 800))}"

ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$ENV" ] || ENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
export PATH="$ENV/bin:$PATH" PYTHONPATH="$REPO/src"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
export HOME=/data/prasann TRITON_CACHE_DIR=/data/prasann/triton_cache FLASHINFER_CACHE_DIR=/data/prasann/flashinfer_cache
mkdir -p "$TRITON_CACHE_DIR" "$FLASHINFER_CACHE_DIR" "$OUTROOT" 2>/dev/null || true
[ -f "$CKPT/config.json" ] || { echo "FATAL: $CKPT has no config.json"; exit 3; }
[ -f "$CKPT/model_and_optim/.metadata" ] || { echo "FATAL: $CKPT/model_and_optim/.metadata missing"; exit 3; }
[ -f "$EVAL_JSONL" ] || { echo "FATAL: missing eval jsonl $EVAL_JSONL"; exit 3; }
echo "=== HOST=$(hostname) TASK=$TASK VARIANT=$VARIANT ARM=$ARM RUNG=$RUNG CKPT=$CKPT PORT=$MASTER_PORT START=$(date '+%F %T') ==="
cd "$REPO"
python -m scripts.eval.ctc_suite.run_rung_eval \
  --task "$TASK" --ckpt "$CKPT" --variant "$VARIANT" --arm "$ARM" --model-scale "$SCALE" \
  --rung-tokens "$RUNG" --eval-jsonl "$EVAL_JSONL" \
  --tokenizer "$TOKENIZER" \
  --doc-start-id "$DOC_START_ID" --doc-end-id "$DOC_END_ID" --eos-token-id "$EOS_ID" \
  ${MAXLEN:+--max-length "$MAXLEN"} \
  --trained-ctx "2k-16k-joint-uniform" \
  --batch-size 1 --master-port "$MASTER_PORT" \
  --out-root "$OUTROOT" --small-eval-ok --nproc "$NGPU"
echo "=== DONE $(date '+%F %T') rc=$? ==="
