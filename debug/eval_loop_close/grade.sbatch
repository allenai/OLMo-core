#!/bin/bash
# The grading half of close_the_loop.sbatch, on the checkpoint that job already trained
# (/data/prasann/ctc_eval_loop/ckpt/step30, which carries config.json and the fingerprint).
#
# Split out after run 3438729 got the fingerprint guard right in both directions and then died in
# DECODING -- `'TorchAttentionBackend' doesn't support KV caching` -- on every arm the guard
# accepted. Re-running the training half to test the decode fix would burn a GPU on a checkpoint
# that is already correct.
#SBATCH --job-name=ctc-eval-grade
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=mooney
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=5:00:00
#SBATCH --output=/data/prasann/joblogs/ctc-eval-grade-%j.log

set -euo pipefail

REPO=/accounts/projects/berkeleynlp/prasann/projects/newolmocore/OLMo-core
WORK=/data/prasann/ctc_eval_loop
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
TOK=$WORK/tok
CKPT=$WORK/ckpt/step30
RES=$WORK/results
DATA=$WORK/data/mathmatch/eval_2k.jsonl

export HOME=$WORK/home
export TMPDIR=$WORK/tmp
export TRITON_CACHE_DIR=$WORK/triton
export HF_HOME=$WORK/hf
export HF_HUB_OFFLINE=1
mkdir -p "$HOME" "$TMPDIR" "$TRITON_CACHE_DIR" "$RES"

echo "=== node $(hostname)  gpus=${CUDA_VISIBLE_DEVICES:-unset}  $(date +%T) ==="
test -f "$CKPT/config.json" || { echo "no config.json in $CKPT"; exit 2; }
cd "$REPO"
export PYTHONPATH=src:ctc/src

set +e
EVAL="$PY debug/eval_loop_close/ctc_eval_mathmatch.py --data $DATA --ckpt $CKPT --tasks mathmatch \
    --backend native --tokenizer $TOK --max-length 4096 --out $RES --overwrite"

run_arm () {  # name, expectation, then ctc-eval args
    local name=$1 expect=$2; shift 2
    echo ""
    echo "############ ARM $name  (expected: $expect)  $(date +%T) ############"
    $EVAL "$@"
    echo "############ ARM $name EXIT=$? ############"
}

run_arm A "runs, produces a number" --attn chunked --query-position after --tag accept
run_arm B "REFUSED: query_position"  --attn chunked --query-position both  --tag rejectqp
run_arm C "REFUSED: chunk_layout"    --attn landmark --query-position after --tag rejectlandmark
run_arm D "runs, records UNVERIFIED" --attn chunked --query-position both --tag ignoreqp \
    --ignore-format-fingerprint --limit 50
run_arm E "runs, produces a number" --attn full --query-position after --tag full

echo ""
echo "=== results written ==="
find "$RES" -maxdepth 1 -name '*.json'
echo "=== DONE $(date +%T) ==="
