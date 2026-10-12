#!/bin/bash
# Three-arm pooled-KV experiment (contradiction n100, Qwen3-4B, LOCAL horton). One sbatch, ARM picks:
#   ARM=full      -> plain causal baseline (accuracy + wall-clock reference)
#   ARM=pooledkv  -> v1 exact-mean pooled-doc-KV attention (matched wall-clock arm)
#   ARM=softtoken -> B1 soft-token compaction (the >=10x speedup arm; needs the -b1 baked base)
#
# NFS-safe (local_cluster.md): --output on node-local /data + /scratch mirror, node-local env, data
# and bases already on horton /data (pooledkv_exp/). See README.md in this folder.
#
#   sbatch --export=ALL,ARM=full,RUN=q4b-pooledkv-full run_q4b_pooledkv_horton.sbatch
#   sbatch --export=ALL,ARM=pooledkv,RUN=q4b-pooledkv-v1 run_q4b_pooledkv_horton.sbatch
#   sbatch --export=ALL,ARM=softtoken,RUN=q4b-pooledkv-b1 run_q4b_pooledkv_horton.sbatch
#SBATCH --job-name=q4b-pooledkv
#SBATCH --account=site
#SBATCH --partition=berkeleynlp
#SBATCH --qos=preemptive
#SBATCH --nodelist=horton
# Default matches our standard NGPU=2 runs; pass --gres at submit for more. An 8-GPU default
# silently eats the whole 8-GPU/user preemptive_high quota per job (QOSMaxGRESPerUser pends).
#SBATCH --gres=gpu:H200:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=/data/prasann/joblogs/q4b_pooledkv_%x_%j.log

set -uo pipefail
MIRROR_DIR=/scratch/users/prasann/pooledkv_logs
mkdir -p "$MIRROR_DIR" 2>/dev/null || true
_SRC_LOG="/data/prasann/joblogs/q4b_pooledkv_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
_DST_LOG="$MIRROR_DIR/train_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
_mirror_log() { cp -f "$_SRC_LOG" "$_DST_LOG" 2>/dev/null || true; }
( while true; do _mirror_log; sleep 30; done ) &
_MIRROR_PID=$!
trap '_mirror_log; kill $_MIRROR_PID 2>/dev/null' EXIT

REPO="${REPO:-$(git -C "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" rev-parse --show-toplevel)}"
NGPU="${NGPU:-8}"
ARM="${ARM:?set ARM=full|pooledkv|softtoken}"
case "$ARM" in
  full|pooledkv|softtoken) ;;
  *) echo "FATAL: unknown ARM='$ARM'"; exit 2 ;;
esac
RUN="${RUN:-q4b-pooledkv-$ARM}"
EPOCHS="${EPOCHS:-3}"
SEQ_LEN="${SEQ_LEN:-6144}"
LR="${LR:-1e-5}"
N_RANDOM="${N_RANDOM:-2}"
SEED="${SEED:-0}"
EXTRA="${EXTRA:-}"
WANDB_GROUP="${WANDB_GROUP:-pooledkv-contra-n100}"
echo "=== HOST=$(hostname) ARM=$ARM RUN=$RUN SEQ_LEN=$SEQ_LEN EPOCHS=$EPOCHS START=$(date '+%F %T') ==="

ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
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
echo "wandb group: https://wandb.ai/prasann-uc-berkeley-electrical-engineering-computer-sciences/memory-networks/groups/${WANDB_GROUP}"

mapfile -t CLEAN < <(nvidia-smi --query-gpu=memory.used,uuid --format=csv,noheader,nounits | awk -F', ' '$1+0 < 2000 {print $2}')
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${CLEAN[*]:0:$NGPU}")
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Base sanity (missing .metadata -> silent train-from-scratch; see the attn_explore template).
if [ "$ARM" = "softtoken" ]; then
  BASE=/data/prasann/pooledkv_exp/q4b-dense-cpt-fixmark-b1/model_and_optim
else
  BASE=/data/prasann/pooledkv_exp/q4b-dense-cpt-fixmark/model_and_optim
fi
if [ ! -f "$BASE/.metadata" ]; then
  echo "FATAL: base checkpoint $BASE missing .metadata"; exit 3
fi
DATA="${DATA:-/data/prasann/pooledkv_exp/contra_n100_qboth_train}"
[ -f "$DATA/metadata.json" ] || { echo "FATAL: data shard $DATA missing"; exit 4; }
[ "$ARM" = "full" ] || [ -f "$DATA/gold_fingerprints.json" ] || { echo "FATAL: gold sidecar missing"; exit 5; }

cd "$REPO"
MASTER_PORT=$((29000 + RANDOM % 1000))
torchrun --nproc_per_node="$NGPU" --master_port="$MASTER_PORT" \
  research/token_routing/pooledkv_train/Qwen3-4B-pooledkv-contra-n100-local.py \
  --arm "$ARM" --run-name "$RUN" --seq-len "$SEQ_LEN" --epochs "$EPOCHS" --lr "$LR" \
  --n-random "$N_RANDOM" --seed "$SEED" --wandb-group "$WANDB_GROUP" \
  $EXTRA $WANDB_FLAG
RC=$?
# Trainer step checkpoints carry ~48GB of fp32 optimizer state each -- only the model-only export
# (model_and_optim + config.json at the run root) is needed for eval. Reclaim the disk (a full
# /data killed a rank mid-save once already), but ONLY if the export exists.
SAVE_DIR=/data/prasann/pooledkv_exp/runs/$RUN
if [ "$RC" -eq 0 ] && [ -f "$SAVE_DIR/model_and_optim/.metadata" ]; then
  rm -rf "$SAVE_DIR"/step* 2>/dev/null
  echo "cleaned trainer step checkpoints under $SAVE_DIR"
fi
echo "=== DONE $(date '+%F %T') rc=$RC ==="
exit $RC
