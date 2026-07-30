#!/bin/bash
# Native (olmo_core) CTC rung eval for the LLAMA family on the LAMBDA cluster.
# A copy of src/scripts/eval/ctc_suite/eval_ctc_lambda.sbatch with the Llama tokenizer + marker ids
# wired in: run_rung_eval.py defaults to the Qwen3.5 ids and the Qwen tokenizer, and a
# wrong-tokenizer eval produces plausible numbers rather than a crash. Kept as a separate file so
# the shared launcher is untouched while sibling families run concurrently.
#
# --batch-size 1 is deliberate: the evaluator's left-pad id is the hardcoded Qwen PAD_TOKEN_ID,
# which is out of range for Llama's 128256-row vocab. At batch-size 1 no pad token ever enters the
# sequence, so it is inert.
#
#   sbatch --gres=gpu:A100:2 -w lambda-hyperplane05 --export=ALL,CKPT=...,TASK=qdmatch,\
#     VARIANT=dense,ARM=full,RUNG=2048,EVAL_JSONL=...,NGPU=2 eval_llama_lambda.sbatch
#SBATCH --job-name=llama-eval-lambda
#SBATCH --partition=lambda
#SBATCH --account=site
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/accounts/projects/sewonm/prasann/ctc_suite/logs/llama_eval_%x_%j.log
set -o pipefail   # deliberately no -u: conda-pack's activate references unset vars

LROOT=/accounts/projects/sewonm/prasann/ctc_suite
REPO=$LROOT/OLMo-core
ENV=/accounts/projects/sewonm/prasann/olmo_test/env

NGPU="${NGPU:-2}"
TASK="${TASK:?set TASK}"; VARIANT="${VARIANT:-dense}"; ARM="${ARM:-full}"
SCALE="${SCALE:-llama3.2-3b}"; RUNG="${RUNG:-2048}"
CKPT="${CKPT:?set CKPT}"; EVAL_JSONL="${EVAL_JSONL:?set EVAL_JSONL}"
OUTROOT="${OUTROOT:-$LROOT/eval_out_llama/${TASK}_${ARM}}"
MAXLEN="${MAXLEN:-$((RUNG + 3072))}"
TOKENIZER="${TOKENIZER:-$LROOT/hf_models/Llama-3.2-3B-marker-tok}"
DOC_START_ID="${DOC_START_ID:-128002}"; DOC_END_ID="${DOC_END_ID:-128003}"; EOS_ID="${EOS_ID:-128001}"

source "$ENV/bin/activate"
conda-unpack 2>/dev/null || true
export PYTHONPATH="$REPO/src"
export TOKENIZERS_PARALLELISM=false PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export LLAMA_MARKER_TOKENIZER="$TOKENIZER"
export HOME=/tmp/prasann
export TRITON_CACHE_DIR=/tmp/prasann/triton_cache
export FLASHINFER_CACHE_DIR=/tmp/prasann/flashinfer_cache
mkdir -p "$TRITON_CACHE_DIR" "$FLASHINFER_CACHE_DIR" "$OUTROOT" 2>/dev/null || true

[ -f "$CKPT/config.json" ] || { echo "FATAL: $CKPT/config.json missing"; exit 3; }
[ -f "$CKPT/model_and_optim/.metadata" ] || { echo "FATAL: $CKPT/model_and_optim/.metadata missing"; exit 3; }
[ -f "$EVAL_JSONL" ] || { echo "FATAL: $EVAL_JSONL missing"; exit 3; }
[ -f "$TOKENIZER/tokenizer.json" ] || { echo "FATAL: tokenizer $TOKENIZER missing"; exit 3; }

MASTER_PORT="${MASTER_PORT:-$((29000 + SLURM_JOB_ID % 900))}"
echo "=== HOST=$(hostname) TASK=$TASK VARIANT=$VARIANT ARM=$ARM RUNG=$RUNG MAXLEN=$MAXLEN PORT=$MASTER_PORT CKPT=$CKPT START=$(date '+%F %T') ==="
cd "$REPO"
python -m scripts.eval.ctc_suite.run_rung_eval \
  --task "$TASK" --ckpt "$CKPT" --variant "$VARIANT" --arm "$ARM" --model-scale "$SCALE" \
  --rung-tokens "$RUNG" --eval-jsonl "$EVAL_JSONL" \
  --tokenizer "$TOKENIZER" \
  --doc-start-id "$DOC_START_ID" --doc-end-id "$DOC_END_ID" --eos-token-id "$EOS_ID" \
  --max-length "$MAXLEN" --master-port "$MASTER_PORT" \
  --trained-ctx "2k-16k-joint-uniform" --batch-size 1 \
  --out-root "$OUTROOT" --small-eval-ok --nproc "$NGPU"
echo "=== DONE $(date '+%F %T') rc=$? ==="
