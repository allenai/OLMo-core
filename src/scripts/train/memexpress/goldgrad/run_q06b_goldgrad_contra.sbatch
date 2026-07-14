#!/bin/bash
# Gold-gradient (O(1)-backward) contradiction-n20 SFT: three arms, one sbatch, SEQUENTIAL on one node.
# All init from the SAME shared dense CPT base + SAME gold-aligned docdense-n20 data; only the backward
# gradient policy (--grad-mode) differs, so the forward/loss is identical across arms.
#
#   full             -> normal document-chunked training (every doc gets gradient). Upper-bound ref.
#   gold_plus_random -> O(1): keep grad for gold contradiction docs + N_RANDOM random docs; detach rest.
#   random_only      -> control: keep |gold|+N_RANDOM RANDOM docs (gold identity hidden), same sparsity.
#
# NFS-safe (see local-train-nfs-log-hang memory): --output on node-local /data, direct PATH (no conda
# activate), PYTHONWARNINGS=ignore, data+workdir staged node-local; base read in place (distcp = reads,
# and staging via cp drops the hidden .metadata -> silent train-from-scratch). Checkpoints saved to
# /scratch so the existing docchunk-contra eval can read them from any node.
#
#   sbatch --export=ALL run_q06b_goldgrad_contra.sbatch                       # all 3 arms, full run
#   sbatch --export=ALL,MAX_STEPS=3,GRAD_MODES=gold_plus_random run_q06b_goldgrad_contra.sbatch  # smoke
#SBATCH --job-name=q06b-goldgrad
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodes=1
#SBATCH --exclude=mcfuzz,sneetches
#SBATCH --gres=gpu:H200:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=06:00:00
# NOTE: --output is node-local /data (NFS-hang-safe, see local-train-nfs-log-hang memory) and must
# pre-exist on the node: a job landing on a fresh node whose /data/prasann/joblogs is missing FAILS in
# ~2s before the script runs (this killed job 3323681 on mcfuzz). Restrict to the H200 nodes that have
# it via --exclude, NOT via --nodelist: -w with two names forces BOTH nodes (16 GPUs) and the job then
# pends forever on the 8-GPU/user QOSMaxGRESPerUser cap.
#SBATCH --output=/data/prasann/joblogs/q06b_goldgrad_%x_%j.log

set -uo pipefail
MIRROR_DIR=/scratch/users/prasann/attn_explore_logs
mkdir -p "$MIRROR_DIR" 2>/dev/null || true
_SRC_LOG="/data/prasann/joblogs/q06b_goldgrad_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
_DST_LOG="$MIRROR_DIR/goldgrad_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
_mirror_log() { cp -f "$_SRC_LOG" "$_DST_LOG" 2>/dev/null || true; }
( while true; do _mirror_log; sleep 30; done ) & _MIRROR_PID=$!
trap '_mirror_log; kill $_MIRROR_PID 2>/dev/null' EXIT

REPO="${REPO:-/accounts/projects/berkeleynlp/prasann/projects/OLMo-core}"
NGPU="${NGPU:-8}"
MAX_STEPS="${MAX_STEPS:-0}"
SEQ_LEN="${SEQ_LEN:-2048}"
EPOCHS="${EPOCHS:-3}"
N_RANDOM="${N_RANDOM:-2}"
GRAD_MODES="${GRAD_MODES:-full gold_plus_random random_only}"
# ATTENTION: default FULL (plain causal). random_doc @ doc_keep_prob=1.0 is provably identical to a
# plain causal mask (verified: 0 causal-allowed positions blocked); we keep the document-chunked code
# path only so box markers / chunk_ids / gold fingerprints / the existing eval work unchanged.
# CROSS_DOC_MODE=chunked + DOC_KEEP_PROB=0 reproduces the pure-chunked control, which floors ALL arms
# (incl. full-grad) at the chance CE (0.486 = format learned, documents guessed) -- do not use it as
# the baseline for gold-grad.
CROSS_DOC_MODE="${CROSS_DOC_MODE:-random_doc}"
DOC_KEEP_PROB="${DOC_KEEP_PROB:-1.0}"
TAG="${TAG:-fa}"      # run-name tag: fa = full attention
NDOCS="${NDOCS:-n20}" # which rung: n20 | n100 (must match DATA_SRC)
N_GOLD="${N_GOLD:-1}" # gold_subsample: # gold docs kept (constant in N -> O(1))
# -fixmark = marker-embedding-REPAIRED base. REQUIRED for any marker-bearing (doc-chunked) shard: on a
# stock Qwen3 base <|box_start|>/<|box_end|> are bit-identical (cos=1.0, never trained), and at 100 docs
# that yields low train CE but ~0 held-out f1. See document-chunked-marker-embeddings.md.
BASE_SRC="${BASE_SRC:-/scratch/users/prasann/cpt_mix_ckpts/q06b-dense-cpt-modelonly-fixmark}"
DATA_SRC="${DATA_SRC:-/scratch/users/prasann/longctx_sft_qwen/contradiction_n20_docdense_nocot_gold}"
SAVE_ROOT="${SAVE_ROOT:-/scratch/users/prasann/olmo_ckpts}"
echo "=== HOST=$(hostname) MODES='$GRAD_MODES' N_RANDOM=$N_RANDOM SEQ_LEN=$SEQ_LEN EPOCHS=$EPOCHS MAX_STEPS=$MAX_STEPS START=$(date '+%F %T') ==="

# ENV_ROOT=scratch forces the /scratch env, which is pinned to the KNOWN-GOOD flash-attn 2.8.2 (2.8.3
# has a varlen-backward bug that SIGSEGVs mid-run -- see the flash-attn-sigsegv-fix memory). The
# node-local /data clone is faster to import from but may carry the buggy build.
ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ "${ENV_ROOT:-data}" = "scratch" ] && ENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$ENV" ] || ENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$REPO/src"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONWARNINGS=ignore
export PYTHONUNBUFFERED=1
if [ -z "${WANDB_API_KEY:-}" ]; then
  WANDB_API_KEY=$(awk '/machine api.wandb.ai/{f=1} f&&/password/{print $2; exit}' "$HOME/.netrc" 2>/dev/null)
  export WANDB_API_KEY
fi
WANDB_FLAG=""
[ -z "${WANDB_API_KEY:-}" ] && WANDB_FLAG="--no-wandb" && echo "WARN: no wandb creds -> --no-wandb"

mapfile -t CLEAN < <(nvidia-smi --query-gpu=memory.used,uuid --format=csv,noheader,nounits | awk -F', ' '$1+0 < 2000 {print $2}')
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${CLEAN[*]:0:$NGPU}")
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

BASE_CKPT="$BASE_SRC/model_and_optim"
if [ ! -f "$BASE_CKPT/.metadata" ]; then
  echo "FATAL: base checkpoint $BASE_CKPT missing .metadata -> would silently train from scratch"; exit 3
fi

# Stage the (tiny) gold data dir node-local.
ROOT=/data/prasann/goldgrad
DATA_LOCAL=$ROOT/data/$(basename "$DATA_SRC")
mkdir -p "$DATA_LOCAL"
cp -un "$DATA_SRC"/* "$DATA_LOCAL"/ 2>/dev/null || cp -n "$DATA_SRC"/* "$DATA_LOCAL"/ 2>/dev/null || true
[ -f "$DATA_LOCAL/gold_fingerprints.json" ] || { echo "FATAL: gold sidecar missing in $DATA_LOCAL"; exit 4; }

cd "$REPO"
MAXARG=""; [ "$MAX_STEPS" -gt 0 ] && MAXARG="--max-steps $MAX_STEPS"

_MODE_IDX=0
for MODE in $GRAD_MODES; do
  case "$MODE" in
    full)             RUN="q06b-goldgrad-${TAG}-${NDOCS}-full" ;;
    gold_plus_random) RUN="q06b-goldgrad-${TAG}-${NDOCS}-gpr${N_RANDOM}" ;;
    gold_subsample)   RUN="q06b-goldgrad-${TAG}-${NDOCS}-gsub${N_GOLD}_${N_RANDOM}" ;;
    random_only)      RUN="q06b-goldgrad-${TAG}-${NDOCS}-rand${N_RANDOM}" ;;
    random_nongold)   RUN="q06b-goldgrad-${TAG}-${NDOCS}-nogold${N_RANDOM}" ;;
    *) echo "unknown MODE=$MODE"; exit 5 ;;
  esac
  WORK_LOCAL=$ROOT/cache-$SLURM_JOB_ID-$MODE; mkdir -p "$WORK_LOCAL"
  SAVE_FOLDER="$SAVE_ROOT/$RUN"
  # Derive the port from the JOB ID, not $RANDOM: co-tenant torchrun jobs on the same node using the
  # same 29000+RANDOM scheme can collide, and colliding ranks rendezvous against the WRONG master and
  # hang until the 30-min DistStoreError timeout (seen on cubbins). Job ids are unique, so this isn't.
  MASTER_PORT=$((29000 + (SLURM_JOB_ID + _MODE_IDX) % 2000))
  _MODE_IDX=$((_MODE_IDX + 1))
  echo "=== [$MODE] RUN=$RUN SAVE=$SAVE_FOLDER $(date '+%F %T') ==="
  torchrun --nproc_per_node="$NGPU" --master_port="$MASTER_PORT" \
    src/scripts/train/memexpress/goldgrad/Qwen3-0.6B-goldgrad-contradiction-n20-SFT-local.py \
    --run-name "$RUN" --base-checkpoint "$BASE_CKPT" \
    --data-dir "$DATA_LOCAL" --work-dir "$WORK_LOCAL" --save-folder "$SAVE_FOLDER" \
    --grad-mode "$MODE" --n-random "$N_RANDOM" --n-gold "$N_GOLD" \
    --cross-doc-mode "$CROSS_DOC_MODE" --doc-keep-prob "$DOC_KEEP_PROB" \
    --seq-len "$SEQ_LEN" --epochs "$EPOCHS" --grad-accum 8 --num-workers 0 \
    --save-checkpoint --wandb-group "goldgrad-contra-n20-${TAG}" $MAXARG $WANDB_FLAG
  echo "=== [$MODE] rc=$? done $(date '+%F %T') ==="
done
echo "=== ALL DONE $(date '+%F %T') ==="
