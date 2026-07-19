#!/bin/bash
# Held-out contradiction eval for the multi-hop gold-routing arms (records/multihop-gold-routing-experiment.md).
# One job, all arms, on the n=50 held-out set. Modelled on eval_q06b_goldgrad_contra.sbatch -- read that
# file's comments too; every trap it documents applies here.
#
# The arms differ only in the TRAINING mask, and the mask is restored from each checkpoint's
# config.json, so this one script scores every arm identically. That symmetry is what makes the
# comparison meaningful -- do not special-case an arm here.
#
#   sbatch --export=ALL eval_q06b_hopgold_contra_n50.sbatch                        # all Stage-1 arms
#   sbatch --export=ALL,RUNS=q06b-hopgold-n50-chunked eval_q06b_hopgold_contra_n50.sbatch
#SBATCH --job-name=q06b-hopgold-eval
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodes=1
#SBATCH --nodelist=mooney
#SBATCH --gres=gpu:H200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/data/prasann/joblogs/q06b_hopgold_eval_%x_%j.log

set -uo pipefail
MIRROR_DIR=/scratch/users/prasann/attn_explore_logs
mkdir -p "$MIRROR_DIR" 2>/dev/null || true
_SRC_LOG="/data/prasann/joblogs/q06b_hopgold_eval_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
_mirror_log() { cp -f "$_SRC_LOG" "$MIRROR_DIR/eval_hopgold_${SLURM_JOB_ID}.log" 2>/dev/null || true; }
( while true; do _mirror_log; sleep 20; done ) & _MIRROR_PID=$!

REPO="${REPO:-/accounts/projects/berkeleynlp/prasann/projects/OLMo-core}"
# Eval code MUST come from this repo, not the /scratch standalone checkout: the standalone still has
# the OLD document wrap (claim label OUTSIDE the chunk = the free-token leak), so a model trained on
# leak-free shards is scored against a prompt layout it never saw -- parse_rate 1.0, f1 0.000, train
# CE ~0.001. Overridable, but don't.
CR_SRC="${CR_SRC:-$REPO/src/corpus_reasoning}"
# Code is in-tree; the eval JSONLs are not (data stays on /scratch). Don't fold this into CR_SRC.
CR_DATA_SRC="${CR_DATA_SRC:-/scratch/users/prasann/corpus-reasoning/data}"
RUNS="${RUNS:-q06b-hopgold-n50-chunked q06b-hopgold-n50-full q06b-hopgold-n50-rand0.25}"
# ⚠ /scratch/users/prasann/olmo_ckpts is a SYMLINK to the NODE-LOCAL /data/prasann/olmo_ckpts, so the
# checkpoints exist ONLY on the node that trained them. Hence --nodelist=mooney above (Stage 1 trained
# there) and reading them in place. If you train elsewhere, re-pin this. A `ls` of the /scratch path
# from the login host resolves against the LOGIN host's /data and reports "not found" -- check on the node.
CKPT_ROOT="${CKPT_ROOT:-/data/prasann/olmo_ckpts}"
NDOCS="${NDOCS:-50}"
CONTRA_FILE="${CONTRA_FILE:-contradiction_eval_pubmed_both_n${NDOCS}_k3.jsonl}"
TOK="${TOK:-Qwen/Qwen3-0.6B}"
# --max-length MUST exceed the PROMPT or the model gets zero generation budget and emits an EMPTY
# string -> f1 0.000 at parse_rate 1.0 for a PERFECTLY trained model. This faked an entire "goldgrad
# doesn't replicate" conclusion once. ⚠ If any arm scores exactly 0.000, DUMP GENERATIONS first.
#
# ⚠ But it is NOT free insurance -- it is the eval's DOMINANT cost. The eval calls
# `prepare_inference_cache(1, max_length)` PER EXAMPLE, so this sizes a fresh KV cache 488 times:
# 28 layers x max_length x 2(K,V) x 1024 x 2B ~= 940MB/example at 8192, ~470MB at 4096. Dropping
# 8192 -> 4096 roughly halves it. Size this to the SHARD, don't pad it "just in case":
#   contra_n50_v2_orig      max_example_len 3151
#   contra_n50_v2_orig_sum10 max_example_len 3411  (summary spans add ~260 tok)
# 4096 clears both with ~700 tokens of headroom. An n=100 shard (~6.1k) needs 8192 -- but then say so.
MAXLEN="${MAXLEN:-4096}"
# no-cot answers are ~30 tokens ("[[3, 22], [4, 24], [5, 21]]"); 256 is slack for the no-EOS rambling
# that --eos-token-id + first-]] truncation are there to cut.
MAXNEW="${MAXNEW:-256}"
NGPU="${NGPU:-2}"
# --variant picks the INFERENCE attention: `dense` = DocumentChunked mask (restored from config.json),
# `full` = plain causal, ignoring the mask. Every Stage-1 arm TRAINED through the document-chunked
# module (chunked / standard / random_doc all set cross_doc_mode), so `dense` is what matches training
# for ALL of them -- including `standard`, whose config.json simply says cross_doc_mode=standard.
# Do not use `full` here: it would silently score the chunked arm without its mask.
VARIANT="${VARIANT:-dense}"
# Data is no-cot -> --cot-mode none. The prefill format MUST match training or the model sees a layout
# it never trained on.
COT_MODE="${COT_MODE:-none}"
# 488 = the ENTIRE n50 eval file. Anything less is a smaller eval and must be flagged inline with its
# error bar (CLAUDE.md §reporting). Binomial SE at 488 is +-0.021 at f1~0.7.
MAX_TEST_SAMPLES="${MAX_TEST_SAMPLES:-488}"
# ⚠ MUST match the SHARD the arm trained on: 0 = a plain docdense shard, 10 = a summary-span shard
# (contra_n50_v2_orig_sum10). The eval renders the prompt itself, so a mismatch does not error -- it
# silently emits a different chunk layout than the model trained on, rebinding every chunk role. The
# bandwidth/relay knobs are NOT passed here: they are attention-config values restored from the
# checkpoint's config.json, which is why one shard serves every rung of the ladder.
SUMMARY_EVERY_K="${SUMMARY_EVERY_K:-0}"
SUMARG=""; [ "$SUMMARY_EVERY_K" -gt 0 ] && SUMARG="--summary-every-k $SUMMARY_EVERY_K"

# ---- stage eval code + data node-local (/scratch read ONCE, not per example) ----
CR=/data/prasann/hopgold/cr_eval
mkdir -p "$CR/data" "$CR/outputs/eval_results"
rsync -a --exclude '__pycache__' "$CR_SRC/" "$CR/" 2>/dev/null
cp -u "$CR_DATA_SRC/$CONTRA_FILE" "$CR/data/" 2>/dev/null
[ -f "$CR/data/$CONTRA_FILE" ] || { echo "FATAL: eval set $CONTRA_FILE not staged from $CR_DATA_SRC"; exit 2; }

_finish() {
  cp -f "$CR"/outputs/eval_results/hopgold-* "$MIRROR_DIR/" 2>/dev/null || true
  _mirror_log; kill $_MIRROR_PID 2>/dev/null
}
trap _finish EXIT

ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$ENV" ] || ENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
export PATH="$ENV/bin:$PATH"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONWARNINGS=ignore
export PYTHONPATH="$REPO/src:$CR"
cd "$CR"

mapfile -t CLEAN < <(nvidia-smi --query-gpu=memory.used,uuid --format=csv,noheader,nounits | awk -F', ' '$1+0 < 2000 {print $2}')
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${CLEAN[*]:0:$NGPU}")
echo "=== HOST=$(hostname) RUNS='$RUNS' EVAL=$CONTRA_FILE VARIANT=$VARIANT MAXLEN=$MAXLEN CUDA=$CUDA_VISIBLE_DEVICES START=$(date '+%F %T') ==="

for RUN in $RUNS; do
  # ⚠ Use the RUN ROOT, not a stepN subdir. The layout is:
  #     <run>/config.json          <- only here
  #     <run>/model_and_optim      <- FINAL, model-only (~2.7G), written LAST
  #     <run>/step250 .. step927   <- periodic FULL checkpoints (model+optim, ~7.6G), NO config.json
  # The eval needs config.json AND model_and_optim in the SAME dir, so only the root works. An earlier
  # version of this script "improved" on the goldgrad template by selecting the highest stepN and every
  # arm died with `FileNotFoundError: .../step927/config.json`. The simple template was correct.
  MODEL_PATH="$CKPT_ROOT/$RUN"
  OUT="outputs/eval_results/hopgold-${RUN}-${VARIANT}_native.json"
  if [ ! -d "$MODEL_PATH/model_and_optim" ]; then
    echo "=== SKIP $RUN: no checkpoint at $MODEL_PATH/model_and_optim ==="; continue
  fi
  echo "=== [$RUN] eval start model=$MODEL_PATH $(date '+%F %T') ==="
  MASTER_PORT=$((29500 + (SLURM_JOB_ID % 1000)))
  torchrun --nproc_per_node="$NGPU" --master_port="$MASTER_PORT" \
    eval/eval_lc_native_docchunk_contra.py \
    --variant "$VARIANT" --model-path "$MODEL_PATH" --out "$OUT" \
    --tokenizer "$TOK" --contra-data "data/$CONTRA_FILE" \
    --cot-mode "$COT_MODE" --contra-max-new-tokens "$MAXNEW" --max-length "$MAXLEN" \
    --max-test-samples "$MAX_TEST_SAMPLES" $SUMARG \
    --per-example-out "outputs/eval_results/hopgold-${RUN}-${VARIANT}_perex.json"
  echo "=== [$RUN] rc=$? -> $OUT  $(date '+%F %T') ==="
  cp -f "$CR"/outputs/eval_results/hopgold-* "$MIRROR_DIR/" 2>/dev/null || true
done
echo "=== ALL EVAL DONE $(date '+%F %T') ==="
