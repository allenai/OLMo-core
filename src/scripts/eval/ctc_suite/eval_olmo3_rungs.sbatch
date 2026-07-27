#!/bin/bash
# CTC-suite OLMo3 arm: run the 2k/4k/8k/16k rung ladder for ONE (task, arm) checkpoint in a single
# 8-GPU job, looping run_rung_eval.py over the rungs (one torchrun per rung, model reloaded each
# time -- that is how the driver works; looping here just avoids four queue waits).
#
# The OLMo/dolma2 marker ids and the patched tokenizer are passed through EXTRA_ARGS-style flags,
# so no evaluator code is OLMo-specific. Node MUST be pinned to wherever the checkpoint's /data
# lives (checkpoints are node-local).
#
#   sbatch --partition=jsteinhardt --qos=preemptive --account=site -w mooney --time=06:00:00 \
#     --export=ALL,CKPT=/data/prasann/ctc_suite/ckpts/ctc-olmo3-7b-contra-full,VARIANT=dense,ARM=full \
#     src/scripts/eval/ctc_suite/eval_olmo3_rungs.sbatch
#SBATCH --job-name=ctc-olmo3-eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:H200:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=/data/prasann/joblogs/ctc_olmo3_eval_%x_%j.log
set -uo pipefail

MIRROR=/scratch/users/prasann/ctc_suite_logs
mkdir -p "$MIRROR" 2>/dev/null || true
_SRC_LOG="/data/prasann/joblogs/ctc_olmo3_eval_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
_DST_LOG="$MIRROR/eval_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
( while true; do cp -f "$_SRC_LOG" "$_DST_LOG" 2>/dev/null; sleep 20; done ) &
_MP=$!
trap 'cp -f "$_SRC_LOG" "$_DST_LOG" 2>/dev/null; kill $_MP 2>/dev/null' EXIT

REPO="${REPO:-/accounts/projects/berkeleynlp/prasann/projects/OLMo-core}"
CKPT="${CKPT:?set CKPT=<ckpt dir with config.json + model_and_optim/>}"
VARIANT="${VARIANT:?set VARIANT=dense|chunked}"
ARM="${ARM:?set ARM=full|chunked-mix}"
TASK="${TASK:-contradiction}"
MODEL_SCALE="${MODEL_SCALE:-olmo3-7b}"
RUNGS="${RUNGS:-2048 4096 8192 16384}"
NGPU="${NGPU:-8}"
EVAL_ROOT="${EVAL_ROOT:-/scratch/users/prasann/ctc_suite_staged/eval_rungs}"
EVAL_DIR_NAME="${EVAL_DIR_NAME:-$TASK}"
# OLMo/dolma2 marker set (RESERVED_IDS["olmo3"]) + the patched tokenizer whose <|extra_id_1|> /
# <|extra_id_2|> slots are renamed to <|box_start|> / <|box_end|>.
TOKENIZER="${TOKENIZER:-/scratch/users/prasann/hf_models/Olmo-3-1025-7B-docchunk}"
DOC_START="${DOC_START:-100266}"
DOC_END="${DOC_END:-100267}"
EOS_ID="${EOS_ID:-100257}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$ENV" ] || ENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$REPO/src"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/data/prasann/triton_cache}"
export HOME=/data/prasann/home_cache
mkdir -p "$TRITON_CACHE_DIR" "$HOME" 2>/dev/null || true

echo "=== HOST=$(hostname) CKPT=$CKPT VARIANT=$VARIANT ARM=$ARM TASK=$TASK RUNGS='$RUNGS' \
START=$(date '+%F %T') ==="
[ -f "$CKPT/config.json" ] || { echo "FATAL: $CKPT/config.json missing"; exit 3; }

cd "$REPO"
RC_ALL=0
for RUNG in $RUNGS; do
  JSONL="$EVAL_ROOT/$EVAL_DIR_NAME/rung_${RUNG}.jsonl"
  if [ ! -f "$JSONL" ]; then echo "SKIP rung $RUNG: missing $JSONL"; RC_ALL=1; continue; fi
  # Unique port per rung: concurrent evals on one node otherwise collide on torchrun's default
  # 29500 and all but one die in ~3s.
  MASTER_PORT=$((29000 + RANDOM % 1000))
  echo "########## rung=$RUNG port=$MASTER_PORT $(date '+%F %T') ##########"
  python -u -m scripts.eval.ctc_suite.run_rung_eval \
    --task "$TASK" --ckpt "$CKPT" --variant "$VARIANT" --arm "$ARM" \
    --rung-tokens "$RUNG" --eval-jsonl "$JSONL" \
    --model-scale "$MODEL_SCALE" --nproc "$NGPU" --master-port "$MASTER_PORT" \
    --tokenizer "$TOKENIZER" \
    --doc-start-id "$DOC_START" --doc-end-id "$DOC_END" --eos-token-id "$EOS_ID" \
    --small-eval-ok $EXTRA_ARGS
  RC=$?
  echo "---- rung $RUNG rc=$RC $(date '+%F %T') ----"
  [ $RC -ne 0 ] && RC_ALL=$RC
done
echo "=== DONE rc=$RC_ALL $(date '+%F %T') ==="
exit $RC_ALL
