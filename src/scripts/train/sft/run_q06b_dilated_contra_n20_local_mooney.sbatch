#!/bin/bash
# Local DILATED-SLIDING-WINDOW ("Hierarchical K") Qwen3-0.6B contradiction-n20 (no-cot) SFT on mooney.
# Sibling of run_q06b_dense_contra_n20_local_mooney.sbatch: identical data/base/optimizer, only the
# attention mask differs (per-layer rotating dilation; K/L/base via DILATED_K/DILATED_L/DILATED_BASE).
#
# Smoke:  sbatch --export=ALL,MAX_STEPS=10,NGPU=2 run_q06b_dilated_contra_n20_local_mooney.sbatch
# Full:   sbatch run_q06b_dilated_contra_n20_local_mooney.sbatch
#SBATCH --job-name=q06b-dilated-contra-local
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=mooney
#SBATCH --gres=gpu:H200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/data/prasann/joblogs/q06b_dilated_contra_local_%j.log

set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
NGPU="${NGPU:-2}"
MAX_STEPS="${MAX_STEPS:-0}"
BASE_CKPT="${BASE_CKPT:-/data/prasann/olmo_ckpts/qwen3_0_6B_base_olmo/model_and_optim}"
RUN_NAME="${RUN_NAME:-q06b-dilated-contra-n20-sft-local}"
DILATED_K="${DILATED_K:-3}"
DILATED_L="${DILATED_L:-3}"
DILATED_BASE="${DILATED_BASE:-2}"
echo "=== HOST=$(hostname) NGPU=$NGPU MAX_STEPS=$MAX_STEPS K=$DILATED_K L=$DILATED_L BASE_M=$DILATED_BASE BASE=$BASE_CKPT RUN=$RUN_NAME START=$(date '+%F %T') ==="

ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$ENV" ] || ENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
# Direct PATH instead of `conda activate` (activate hangs for minutes on NFS lock contention when
# multiple jobs start concurrently; the env's bin on PATH is all torchrun needs).
export PATH="$ENV/bin:$PATH"

export PYTHONPATH="$REPO/src"
export TOKENIZERS_PARALLELISM=false
# The HF cache is warm; never hit the hub (blocked/slow egress from compute nodes causes silent
# multi-minute retry stalls in dataloader workers).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# Silence per-process FutureWarning storm + keep the --output log on /data (never /accounts NFS) to
# avoid the nfs_wait_bit_killable stall where concurrent NFS log appends deadlock ranks before step 1.
export PYTHONWARNINGS=ignore
if [ -z "${WANDB_API_KEY:-}" ]; then
  WANDB_API_KEY=$(awk '/machine api.wandb.ai/{f=1} f&&/password/{print $2; exit}' "$HOME/.netrc" 2>/dev/null)
  export WANDB_API_KEY
fi
WANDB_FLAG=""
[ -z "${WANDB_API_KEY:-}" ] && WANDB_FLAG="--no-wandb" && echo "WARN: no wandb creds -> --no-wandb"

mapfile -t CLEAN < <(nvidia-smi --query-gpu=memory.used,uuid --format=csv,noheader,nounits | awk -F', ' '$1+0 < 2000 {print $2}')
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${CLEAN[*]:0:$NGPU}")
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# Stage the (tiny) data + a FRESH work dir on node-local /data: shared-NFS work dirs accumulate
# stale NFSv4 locks from killed jobs (ranks then hang in nfs_wait_bit_killable before step 1).
LOCAL_ROOT=/data/prasann/contra_n20_smoke
DATA_LOCAL=$LOCAL_ROOT/data
mkdir -p "$DATA_LOCAL"
cp -u /scratch/users/prasann/longctx_sft_qwen/contradiction_n20/* "$DATA_LOCAL"/
WORK_LOCAL=$LOCAL_ROOT/cache-$SLURM_JOB_ID
mkdir -p "$WORK_LOCAL"
SAVE_LOCAL=$LOCAL_ROOT/ckpts/$RUN_NAME
mkdir -p "$SAVE_LOCAL"

cd "$REPO"
MAXARG=""; [ "$MAX_STEPS" -gt 0 ] && MAXARG="--max-steps $MAX_STEPS"
MASTER_PORT=$((29000 + RANDOM % 1000))
torchrun --nproc_per_node="$NGPU" --master_port="$MASTER_PORT" \
  src/scripts/train/sft/Qwen3-0.6B-dilated-contradiction-n20-SFT-local.py \
  --run-name "$RUN_NAME" --base-checkpoint "$BASE_CKPT" \
  --data-path "$DATA_LOCAL" --work-dir "$WORK_LOCAL" --save-folder "$SAVE_LOCAL" --num-workers 0 \
  --dilated-k "$DILATED_K" --dilated-num-configs "$DILATED_L" --dilated-base "$DILATED_BASE" \
  --no-compile $MAXARG $WANDB_FLAG
echo "=== DONE $(date '+%F %T')  rc=$? ==="
