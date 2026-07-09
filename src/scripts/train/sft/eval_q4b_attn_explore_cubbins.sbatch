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
CR_SRC=/scratch/users/prasann/corpus-reasoning
MODE="${MODE:?set MODE=full|docchunk|probe}"
RUN="${RUN:?set RUN (matches the train run name)}"
NDOCS="${NDOCS:-100}"
MODEL_PATH="${MODEL_PATH:-/data/prasann/attn_explore/ckpts/$RUN}"
OUT_NAME="${OUT_NAME:-attn-explore-$RUN}"
CONTRA_FILE="${CONTRA_FILE:-contradiction_eval_pubmed_both_n${NDOCS}_k3.jsonl}"
TOK="${TOK:-Qwen/Qwen3-0.6B}"   # same tokenizer.json across Qwen3 sizes; matches the training shards
MAXLEN="${MAXLEN:-8192}"
MAXNEW="${MAXNEW:-256}"
TRAIN_SHARD="${TRAIN_SHARD:-/data/prasann/attn_explore/data/contradiction_n${NDOCS}_docdense_nocot}"

# ---- stage eval code + data to node-local /data (idempotent; /scratch read ONCE here) ----
CR=/data/prasann/attn_explore/cr_eval
mkdir -p "$CR/data" "$CR/outputs/eval_results"
rsync -a --exclude '__pycache__' "$CR_SRC/scripts" "$CR/" 2>/dev/null
cp -u "$CR_SRC/data/$CONTRA_FILE" "$CR/data/" 2>/dev/null
cp -u "$REPO/src/scripts/train/sft/probe_train_memorization.py" "$CR/" 2>/dev/null || true
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
export PYTHONPATH="$REPO/src:$CR"
cd "$CR"

mapfile -t CLEAN < <(nvidia-smi --query-gpu=memory.used,uuid --format=csv,noheader,nounits | awk -F', ' '$1+0 < 2000 {print $2}')
echo "HOST=$(hostname) MODE=$MODE RUN=$RUN MODEL=$MODEL_PATH DATA=$CONTRA_DATA START=$(date '+%F %T')"
[ -d "$MODEL_PATH/model_and_optim" ] || { echo "MISSING checkpoint $MODEL_PATH/model_and_optim"; exit 2; }

OUT="outputs/eval_results/${OUT_NAME}_native.json"
if [ "$MODE" = "full" ]; then
  export CUDA_VISIBLE_DEVICES="${CLEAN[0]}"
  python -u scripts/eval/eval_lc_native.py \
    --model-path "$MODEL_PATH" --out "$OUT" \
    --tokenizer "$TOK" --max-length "$MAXLEN" --contra-max-new-tokens "$MAXNEW" \
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
    scripts/eval/eval_lc_native_docchunk_contra.py \
    --variant dense --model-path "$MODEL_PATH" --out "$OUT" \
    --tokenizer "$TOK" --contra-data "$CONTRA_DATA" \
    --cot-mode none --contra-max-new-tokens "$MAXNEW" --max-length "$MAXLEN" \
    --max-test-samples 100
fi
RC=$?
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "DONE $(date '+%T') rc=$RC -> $OUT"
