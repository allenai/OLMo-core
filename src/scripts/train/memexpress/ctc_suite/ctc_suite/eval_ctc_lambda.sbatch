#!/bin/bash
# Native (olmo_core) CTC rung eval on the LAMBDA cluster -- the counterpart to
# debug/ctc_vllm_validation/eval_ctc_native.sbatch, which is Berkeley-only (it hardcodes the
# jsteinhardt partition, the /accounts repo path, and /data caches that do not exist here).
#
# Lambda differences that matter (see lambda_cluster.md):
#   * repo + env are the rsync'd LROOT copies; env is conda-pack'd, activated with
#     `source $ENV/bin/activate` (and NO `set -u` -- it dies on CONDA_PREFIX).
#   * HF_HOME must point at the populated cache. Lambda is air-gapped, so with HF_HUB_OFFLINE=1 a
#     tokenizer cache miss is fatal rather than a download. This is exactly what silently made eval
#     "impossible" on lambda before.
#   * checkpoints live on node-local /tmp, so the eval MUST be node-pinned (-w) to the node that
#     trained the run. /data is NOT writable on lambda.
#
# Env knobs: CKPT TASK VARIANT ARM SCALE RUNG EVAL_JSONL NGPU MAXLEN OUTROOT
#
#   sbatch --partition=lambda --account=site --qos=preemptive_high --gres=gpu:A100:2 \
#     -w lambda-hyperplane03 --export=ALL,CKPT=/tmp/prasann/ctc_suite/ckpts/<run>,TASK=cycle,\
# VARIANT=dense,ARM=full,RUNG=2048,EVAL_JSONL=$LROOT/eval_rungs/cycle/rung_2048.jsonl,NGPU=2 \
#     src/scripts/eval/ctc_suite/eval_ctc_lambda.sbatch
#
#SBATCH --job-name=ctc-eval-lambda
#SBATCH --partition=lambda
#SBATCH --account=site
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --output=/accounts/projects/sewonm/prasann/ctc_suite/logs/ctc_eval_%x_%j.log
set -o pipefail   # deliberately no -u: conda-pack's activate references unset vars

LROOT=/accounts/projects/sewonm/prasann/ctc_suite
REPO=$LROOT/OLMo-core
ENV=/accounts/projects/sewonm/prasann/olmo_test/env

NGPU="${NGPU:-2}"
TASK="${TASK:?set TASK}"; VARIANT="${VARIANT:-dense}"; ARM="${ARM:-full}"
SCALE="${SCALE:-qwen3.5-4b}"; RUNG="${RUNG:-2048}"
CKPT="${CKPT:?set CKPT}"; EVAL_JSONL="${EVAL_JSONL:?set EVAL_JSONL}"
OUTROOT="${OUTROOT:-$LROOT/eval_out/${TASK}_${ARM}}"
# Decode budget must exceed the rung's PROMPT length or the generation comes back EMPTY and a
# perfect model scores 0.000 at parse_rate 1.0 (this has faked a "result" before). Default to 2x the
# rung rather than something snug.
MAXLEN="${MAXLEN:-$((RUNG * 2))}"

source "$ENV/bin/activate"
conda-unpack 2>/dev/null || true
export PYTHONPATH="$REPO/src"
export TOKENIZERS_PARALLELISM=false PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME_OVERRIDE:-/accounts/projects/sewonm/prasann/.cache/huggingface}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Node-local JIT caches. NEVER put these on NFS -- concurrent evals deadlock in
# nfs_wait_bit_killable and hang with no compiler running.
export HOME=/tmp/prasann
export TRITON_CACHE_DIR=/tmp/prasann/triton_cache
export FLASHINFER_CACHE_DIR=/tmp/prasann/flashinfer_cache
mkdir -p "$TRITON_CACHE_DIR" "$FLASHINFER_CACHE_DIR" "$OUTROOT" 2>/dev/null || true

[ -f "$CKPT/config.json" ] || { echo "FATAL: $CKPT/config.json missing"; exit 3; }
[ -f "$CKPT/model_and_optim/.metadata" ] || { echo "FATAL: $CKPT/model_and_optim/.metadata missing"; exit 3; }
[ -f "$EVAL_JSONL" ] || { echo "FATAL: $EVAL_JSONL missing"; exit 3; }

echo "=== HOST=$(hostname) TASK=$TASK VARIANT=$VARIANT ARM=$ARM RUNG=$RUNG MAXLEN=$MAXLEN CKPT=$CKPT START=$(date '+%F %T') ==="
cd "$REPO"
python -m scripts.eval.ctc_suite.run_rung_eval \
  --task "$TASK" --ckpt "$CKPT" --variant "$VARIANT" --arm "$ARM" --model-scale "$SCALE" \
  --rung-tokens "$RUNG" --eval-jsonl "$EVAL_JSONL" \
  --max-length "$MAXLEN" \
  --out-root "$OUTROOT" --small-eval-ok --nproc "$NGPU"
echo "=== DONE $(date '+%F %T') rc=$? ==="
