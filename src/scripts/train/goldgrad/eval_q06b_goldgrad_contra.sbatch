#!/bin/bash
# Held-out contradiction eval for the gold-gradient (O(1)-backward) n20 arms. One job, all 3 arms.
#
# Gold-grad only changes the TRAINING backward pass -- the saved checkpoints are ordinary
# document-chunked models -- so this is the EXISTING docchunk-contra eval, unchanged and unaware of
# gold. Every arm is scored the same way on the same held-out set, which is what makes the F1
# comparison (full vs gold+2random vs random-only) meaningful.
#
# nocot data -> --cot-mode none (prefill format must match training).
#
#   sbatch --export=ALL eval_q06b_goldgrad_contra.sbatch                       # all 3 arms
#   sbatch --export=ALL,RUNS=q06b-goldgrad-contra-n20-gpr2 eval_q06b_goldgrad_contra.sbatch
#SBATCH --job-name=q06b-goldgrad-eval
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodes=1
#SBATCH --exclude=mcfuzz,sneetches
#SBATCH --gres=gpu:H200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
# --output must be a node-local /data dir that PRE-EXISTS on the node (NFS-hang-safe); see the train
# sbatch for why this is --exclude and not --nodelist.
#SBATCH --output=/data/prasann/joblogs/q06b_goldgrad_eval_%x_%j.log

set -uo pipefail
MIRROR_DIR=/scratch/users/prasann/attn_explore_logs
mkdir -p "$MIRROR_DIR" 2>/dev/null || true
_SRC_LOG="/data/prasann/joblogs/q06b_goldgrad_eval_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
_mirror_log() { cp -f "$_SRC_LOG" "$MIRROR_DIR/eval_goldgrad_${SLURM_JOB_ID}.log" 2>/dev/null || true; }
( while true; do _mirror_log; sleep 20; done ) & _MIRROR_PID=$!

REPO="${REPO:-/accounts/projects/berkeleynlp/prasann/projects/OLMo-core}"
CR_SRC=/scratch/users/prasann/corpus-reasoning
N_RANDOM="${N_RANDOM:-2}"
TAG="${TAG:-fa}"   # must match the train run's TAG (fa = full attention)
RUNS="${RUNS:-q06b-goldgrad-${TAG}-n20-full q06b-goldgrad-${TAG}-n20-gpr${N_RANDOM} q06b-goldgrad-${TAG}-n20-rand${N_RANDOM}}"
# NOTE: /scratch/users/prasann/olmo_ckpts is a SYMLINK to the node-local /data/prasann/olmo_ckpts, so
# the checkpoints only exist on the node that trained them -> pin this eval to that node (-w) and read
# them in place (also what the avoid-/scratch-for-big-blobs rule wants).
CKPT_ROOT="${CKPT_ROOT:-/data/prasann/olmo_ckpts}"
NDOCS="${NDOCS:-20}"
CONTRA_FILE="${CONTRA_FILE:-contradiction_eval_pubmed_both_n${NDOCS}_k3.jsonl}"
TOK="${TOK:-Qwen/Qwen3-0.6B}"
MAXLEN="${MAXLEN:-4096}"
MAXNEW="${MAXNEW:-256}"
NGPU="${NGPU:-2}"

# ---- stage eval code + data node-local (/scratch read ONCE) ----
CR=/data/prasann/goldgrad/cr_eval
mkdir -p "$CR/data" "$CR/outputs/eval_results"
rsync -a --exclude '__pycache__' "$CR_SRC/scripts" "$CR/" 2>/dev/null
cp -u "$CR_SRC/data/$CONTRA_FILE" "$CR/data/" 2>/dev/null
[ -f "$CR/data/$CONTRA_FILE" ] || { echo "FATAL: eval set $CONTRA_FILE not staged"; exit 2; }

_finish() {
  cp -f "$CR"/outputs/eval_results/goldgrad-* "$MIRROR_DIR/" 2>/dev/null || true
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
echo "=== HOST=$(hostname) RUNS='$RUNS' EVAL=$CONTRA_FILE CUDA=$CUDA_VISIBLE_DEVICES START=$(date '+%F %T') ==="

for RUN in $RUNS; do
  MODEL_PATH="$CKPT_ROOT/$RUN"
  OUT="outputs/eval_results/goldgrad-${RUN}_native.json"
  if [ ! -d "$MODEL_PATH/model_and_optim" ]; then
    echo "=== SKIP $RUN: no checkpoint at $MODEL_PATH/model_and_optim ==="; continue
  fi
  echo "=== [$RUN] eval start $(date '+%F %T') ==="
  MASTER_PORT=$((29500 + RANDOM % 1000))
  torchrun --nproc_per_node="$NGPU" --master_port="$MASTER_PORT" \
    scripts/eval/eval_lc_native_docchunk_contra.py \
    --variant dense --model-path "$MODEL_PATH" --out "$OUT" \
    --tokenizer "$TOK" --contra-data "data/$CONTRA_FILE" \
    --cot-mode none --contra-max-new-tokens "$MAXNEW" --max-length "$MAXLEN" \
    --max-test-samples "${MAX_TEST_SAMPLES:-488}"
  echo "=== [$RUN] rc=$? -> $OUT  $(date '+%F %T') ==="
  cp -f "$CR/outputs/eval_results/goldgrad-${RUN}"* "$MIRROR_DIR/" 2>/dev/null || true
done
echo "=== ALL EVAL DONE $(date '+%F %T') ==="
