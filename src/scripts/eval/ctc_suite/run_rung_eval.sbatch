#!/bin/bash
# CTC-suite per-rung eval launcher: one job = one (checkpoint, arm) call into
# run_rung_eval.py (src/scripts/eval/ctc_suite/run_rung_eval.py), which wraps the proven native
# evaluator via torchrun. Mirrors run_ctc_local.sbatch's node-local-first conventions: pin the node
# that holds the checkpoint (--nodelist), read ckpt + eval data from /data (never /scratch for
# job I/O), log to /data/prasann/joblogs, write only the few-KB result JSONs to the repo (/accounts,
# shared NFS -- fine for small files).
#
# Knobs (env): CKPT (required) VARIANT (required: dense|chunked) ARM (required) TASK (default
#              contradiction) RUNG_TOKENS (default 2048) MODEL_SCALE (default qwen3.5-0.8b)
#              EVAL_JSONL (default the n20 contradiction held-out set staged on cubbins /data)
#              EXTRA_ARGS (verbatim passthrough, e.g. --small-eval-ok is already default-on below)
#
#   sbatch --nodelist=cubbins --export=ALL,CKPT=/data/prasann/ctc_suite/ckpts/ctc-t0-contra-n20-full,\
#     VARIANT=dense,ARM=full src/scripts/eval/ctc_suite/run_rung_eval.sbatch
#SBATCH --job-name=ctc-rung-eval
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=00:30:00
#SBATCH --output=/data/prasann/joblogs/ctc_rung_eval_%x_%j.log
set -uo pipefail

REPO="${REPO:-/accounts/projects/berkeleynlp/prasann/projects/OLMo-core}"
CKPT="${CKPT:?set CKPT=<distcp step dir with config.json + model_and_optim/>}"
VARIANT="${VARIANT:?set VARIANT=dense|chunked (driver semantics: dense=full-attn eval, chunked=doc-chunked-mask eval)}"
ARM="${ARM:?set ARM=<§8 arm label, e.g. full|chunked|chunked-mix>}"
TASK="${TASK:-contradiction}"
RUNG_TOKENS="${RUNG_TOKENS:-2048}"
MODEL_SCALE="${MODEL_SCALE:-qwen3.5-0.8b}"
EVAL_JSONL="${EVAL_JSONL:-/data/prasann/ctc_suite_data/eval_rungs/contradiction_eval_pubmed_both_n20_k3.jsonl}"
NGPU="${NGPU:-8}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

echo "=== HOST=$(hostname) CKPT=$CKPT VARIANT=$VARIANT ARM=$ARM TASK=$TASK RUNG_TOKENS=$RUNG_TOKENS START=$(date '+%F %T') ==="

ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$ENV" ] || ENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$REPO/src"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONWARNINGS=ignore
export PYTHONUNBUFFERED=1

MASTER_PORT=$((29000 + RANDOM % 1000))

cd "$REPO"
python -u -m scripts.eval.ctc_suite.run_rung_eval \
  --task "$TASK" \
  --ckpt "$CKPT" \
  --variant "$VARIANT" \
  --rung-tokens "$RUNG_TOKENS" \
  --eval-jsonl "$EVAL_JSONL" \
  --arm "$ARM" \
  --model-scale "$MODEL_SCALE" \
  --nproc "$NGPU" \
  --master-port "$MASTER_PORT" \
  --small-eval-ok \
  $EXTRA_ARGS
RC=$?
echo "=== DONE $(date '+%F %T')  rc=$RC ==="
exit $RC
