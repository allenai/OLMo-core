#!/bin/bash
# Native HELD-OUT contradiction eval for the 4B attention-pattern exploration (LOCAL cubbins).
# Each model is scored WITH ITS TRAINING MASK on the held-out n100 contra eval set:
#   MODE=full    -> eval_lc_native.py (plain full attention, plain prompt; 1 GPU, batched)
#   MODE=docchunk-> eval_lc_native_docchunk_contra.py --variant dense (box-marker chunked prefill;
#                   the checkpoint's config.json drives chunked/hierarchical_dilated/random_doc)
#   MODE=probe   -> probe_train_memorization.py (feed the EXACT training-npy prompt bytes; checks
#                   the docchunk eval machinery reproduces memorized answers -> validates the eval)
# nocot models -> --cot-mode none (prefill format must match training).
#
# NFS-safe: ALL job I/O on node-local /data (corpus-reasoning eval code + eval jsonl + probe script
# staged there; outputs + logs written there); small results mirrored to /scratch on exit only.
#
#   sbatch --export=ALL,MODE=full,RUN=q4b-full-c100b eval_q4b_attn_explore_cubbins.sbatch
#   sbatch --export=ALL,MODE=docchunk,RUN=q4b-hier-n4m2-c100 eval_q4b_attn_explore_cubbins.sbatch
#SBATCH --job-name=q4b-attn-eval
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive
#SBATCH --nodelist=cubbins
#SBATCH --gres=gpu:H200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/data/prasann/joblogs/q4b_attn_eval_%x_%j.log

set -uo pipefail
MIRROR_DIR=/scratch/users/prasann/attn_explore_logs
mkdir -p "$MIRROR_DIR" 2>/dev/null || true
_SRC_LOG="/data/prasann/joblogs/q4b_attn_eval_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
_mirror_log() { cp -f "$_SRC_LOG" "$MIRROR_DIR/eval_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>/dev/null || true; }
( while true; do _mirror_log; sleep 20; done ) &
_MIRROR_PID=$!

REPO="${REPO:-/accounts/projects/berkeleynlp/prasann/projects/OLMo-core}"
# The eval harness now lives IN THIS REPO (src/corpus_reasoning), ported from the standalone
# corpus-reasoning repo -- one tree, one git history, no /scratch checkout to drift against.
# Only the large corpora stay on /scratch (see CR_DATA).
CR_SRC="${CR_SRC:-$REPO/src/corpus_reasoning}"
CR_DATA="${CR_DATA:-/scratch/users/prasann/corpus-reasoning/data}"
MODE="${MODE:?set MODE=full|docchunk|probe}"
RUN="${RUN:?set RUN (matches the train run name)}"
NDOCS="${NDOCS:-100}"
MODEL_PATH="${MODEL_PATH:-/data/prasann/attn_explore/ckpts/$RUN}"
OUT_NAME="${OUT_NAME:-attn-explore-$RUN}"
CONTRA_FILE="${CONTRA_FILE:-contradiction_eval_pubmed_both_n${NDOCS}_k3.jsonl}"
TOK="${TOK:-Qwen/Qwen3-0.6B}"   # same tokenizer.json across Qwen3 sizes; matches the training shards
MAXLEN="${MAXLEN:-8192}"
# Eval-set size (NOT `n` -- `n` means CORPUS size, e.g. the n100 in contradiction_..._n100_k3).
# Default = 488 = the ENTIRE held-out contradiction set. Never shrink this below 500 without
# saying so next to every number it produces: a small eval turns noise into apparent findings.
MAX_TEST_SAMPLES="${MAX_TEST_SAMPLES:-488}"
MAXNEW="${MAXNEW:-256}"
# Prompt-layout knobs -- these MUST equal the values the training shard was built with, or the
# model is scored on a prompt layout it never saw. FREE_PAD adds FREE tokens (which attend the
# whole context); REP_DOC repeats each document's text inside its chunk.
FREE_PAD="${FREE_PAD:-0}"
REP_DOC="${REP_DOC:-1}"
TRAIN_SHARD="${TRAIN_SHARD:-/data/prasann/attn_explore/data/contradiction_n${NDOCS}_docdense_nocot}"

# ---- stage eval code + data to node-local /data (idempotent; NFS/scratch read ONCE here) ----
CR=/data/prasann/attn_explore/cr_eval
mkdir -p "$CR/data" "$CR/outputs/eval_results"
rsync -a --exclude '__pycache__' "$CR_SRC/" "$CR/corpus_reasoning/" 2>/dev/null
cp -u "$CR_DATA/$CONTRA_FILE" "$CR/data/" 2>/dev/null
cp -u "$REPO/src/scripts/train/memexpress/attn_explore/probe_train_memorization.py" "$CR/" 2>/dev/null || true
CONTRA_DATA="data/$CONTRA_FILE"

_mirror_out() {
  cp -f "$CR/outputs/eval_results/${OUT_NAME}"* "$MIRROR_DIR/" 2>/dev/null || true
  _mirror_log; kill $_MIRROR_PID 2>/dev/null
}
trap _mirror_out EXIT

ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$ENV" ] || ENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
export PATH="$ENV/bin:$PATH"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONWARNINGS=ignore
# $CR first: the node-local staged corpus_reasoning wins over the NFS copy in $REPO/src
export PYTHONPATH="$CR:$REPO/src"
cd "$CR"

mapfile -t CLEAN < <(nvidia-smi --query-gpu=memory.used,uuid --format=csv,noheader,nounits | awk -F', ' '$1+0 < 2000 {print $2}')
echo "HOST=$(hostname) MODE=$MODE RUN=$RUN MODEL=$MODEL_PATH DATA=$CONTRA_DATA START=$(date '+%F %T')"
[ -d "$MODEL_PATH/model_and_optim" ] || { echo "MISSING checkpoint $MODEL_PATH/model_and_optim"; exit 2; }

OUT="outputs/eval_results/${OUT_NAME}_native.json"
if [ "$MODE" = "full" ]; then
  export CUDA_VISIBLE_DEVICES="${CLEAN[0]}"
  python -u -m corpus_reasoning.eval.eval_lc_native \
    --model-path "$MODEL_PATH" --out "$OUT" \
    --tokenizer "$TOK" --max-length "$MAXLEN" --contra-max-new-tokens "$MAXNEW" \
    --eos-token-id 151643 --max-test-samples "${MAX_TEST_SAMPLES:-488}" \
    --skip-ruler --contra-data "$CONTRA_DATA" --nq-data /nonexistent_skip_nq --batch-size 16
elif [ "$MODE" = "probe" ]; then
  export CUDA_VISIBLE_DEVICES="${CLEAN[0]}"
  DENSE_FLAG=""; [ "${PROBE_DENSE:-0}" = "1" ] && DENSE_FLAG="--dense"
  python -u probe_train_memorization.py \
    --model-path "$MODEL_PATH" --data-dir "$TRAIN_SHARD" --tokenizer "$TOK" \
    --n "${PROBE_N:-8}" --max-new-tokens "$MAXNEW" --max-length "$MAXLEN" $DENSE_FLAG \
    --out "outputs/eval_results/${OUT_NAME}_probe.json"
else
  NGPU="${NGPU:-2}"
  export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${CLEAN[*]:0:$NGPU}")
  MASTER_PORT=$((29500 + RANDOM % 1000))
  torchrun --nproc_per_node="$NGPU" --master_port="$MASTER_PORT" \
    -m corpus_reasoning.eval.eval_lc_native_docchunk_contra \
    --variant "${EVAL_VARIANT:-dense}" --model-path "$MODEL_PATH" --out "$OUT" \
    --tokenizer "$TOK" --contra-data "$CONTRA_DATA" \
    --cot-mode none --contra-max-new-tokens "$MAXNEW" --max-length "$MAXLEN" \
    --free-pad-repeat "${FREE_PAD:-0}" --repeat-doc-text "${REP_DOC:-1}" \
    --max-test-samples "${MAX_TEST_SAMPLES:-488}"
fi
RC=$?
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "DONE $(date '+%T') rc=$RC -> $OUT"
