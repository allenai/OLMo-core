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
# Eval code MUST come from this repo, not the /scratch standalone checkout. The standalone still has
# the OLD document wrap (claim label left OUTSIDE the chunk = the free-token leak), so a model trained
# on the leak-free shards is scored against a prompt layout it never saw: it emits well-formed output
# (parse_rate 1.0) and f1 collapses to 0.000 while its train CE is ~0.001. Overridable, but don't.
CR_SRC="${CR_SRC:-$REPO/src/corpus_reasoning}"
# ...but the eval JSONLs are NOT in-tree (code is in-tree, data stays on /scratch), so the eval set has
# to be staged from a separate root. Don't fold this back into CR_SRC.
CR_DATA_SRC="${CR_DATA_SRC:-/scratch/users/prasann/corpus-reasoning/data}"
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
# --max-length MUST exceed the PROMPT length or the model gets 0 generation budget and emits an EMPTY
# string -> f1 0.000 at parse_rate 1.0 for a perfectly-trained model. The n100 contradiction prompt is
# ~6144 tokens, so 4096 (fine for n20) SILENTLY truncates it to zero headroom and every arm scores
# 0.000. Scale the default with the rung; override MAXLEN only to go bigger.
if [ -z "${MAXLEN:-}" ]; then case "$NDOCS" in 100|n100) MAXLEN=8192 ;; *) MAXLEN=4096 ;; esac; fi
MAXNEW="${MAXNEW:-256}"
NGPU="${NGPU:-2}"
# --variant selects the INFERENCE attention: `full` = plain causal (no mask), `dense` = DocumentChunked
# mask. The goldgrad arms train plain-causal (random_doc @ doc_keep_prob=1.0). Historically the working
# n100 evals used `dense` (gg3 arms) and `full` (famark); an earlier "variant flip" (one worked, the
# other gave 0.000) was NOT a real attention effect -- it was the MAXLEN truncation above hitting one
# batch of runs. With MAXLEN correct, re-check which variant matches training before trusting a compare.
VARIANT="${VARIANT:-full}"

# ---- stage eval code + data node-local (/scratch read ONCE) ----
CR=/data/prasann/goldgrad/cr_eval
mkdir -p "$CR/data" "$CR/outputs/eval_results"
rsync -a --exclude '__pycache__' "$CR_SRC/scripts" "$CR/" 2>/dev/null
cp -u "$CR_DATA_SRC/$CONTRA_FILE" "$CR/data/" 2>/dev/null
[ -f "$CR/data/$CONTRA_FILE" ] || { echo "FATAL: eval set $CONTRA_FILE not staged from $CR_DATA_SRC"; exit 2; }

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
  # Tag the output with the variant so a `full` re-score does not clobber an earlier `dense` one.
  OUT="outputs/eval_results/goldgrad-${RUN}-${VARIANT}_native.json"
  if [ ! -d "$MODEL_PATH/model_and_optim" ]; then
    echo "=== SKIP $RUN: no checkpoint at $MODEL_PATH/model_and_optim ==="; continue
  fi
  echo "=== [$RUN] eval start $(date '+%F %T') ==="
  MASTER_PORT=$((29500 + RANDOM % 1000))
  torchrun --nproc_per_node="$NGPU" --master_port="$MASTER_PORT" \
    scripts/eval/eval_lc_native_docchunk_contra.py \
    --variant "$VARIANT" --model-path "$MODEL_PATH" --out "$OUT" \
    --tokenizer "$TOK" --contra-data "data/$CONTRA_FILE" \
    --cot-mode none --contra-max-new-tokens "$MAXNEW" --max-length "$MAXLEN" \
    --max-test-samples "${MAX_TEST_SAMPLES:-488}"
  echo "=== [$RUN] rc=$? -> $OUT  $(date '+%F %T') ==="
  cp -f "$CR/outputs/eval_results/goldgrad-${RUN}"* "$MIRROR_DIR/" 2>/dev/null || true
done
echo "=== ALL EVAL DONE $(date '+%F %T') ==="
