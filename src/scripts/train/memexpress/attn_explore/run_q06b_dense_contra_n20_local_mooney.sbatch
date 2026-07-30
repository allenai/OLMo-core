#!/bin/bash
# Local DENSE Qwen3-0.6B contradiction-n20 (no-cot) SFT on mooney. Sibling of
# run_q06b_fast_contra_n20_local_mooney.sbatch. Uses THIS tree's olmo_core, the n20 nocot shards on
# /scratch, and (by default) the RAW Qwen3-0.6B base distcp on mooney /data -- valid for dense since
# dense never uses the landmark token. Override BASE_CKPT for a proper 64k dense base once transferred.
#
# Smoke:  sbatch --export=ALL,MAX_STEPS=10,NGPU=2 run_q06b_dense_contra_n20_local_mooney.sbatch
# Full:   sbatch run_q06b_dense_contra_n20_local_mooney.sbatch
#SBATCH --job-name=q06b-dense-contra-local
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=mooney
#SBATCH --gres=gpu:H200:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=/data/prasann/joblogs/q06b_dense_contra_local_%j.log

set -uo pipefail
# Env conventions (fast env, offline flags, wandb, PYTHONWARNINGS, MASTER_PORT, gpu/workdir
# helpers) all live in local_env.sh -- see local_cluster.md.
source /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/src/scripts/local_env.sh
NGPU="${NGPU:-8}"
MAX_STEPS="${MAX_STEPS:-0}"
BASE_CKPT="${BASE_CKPT:-/data/prasann/olmo_ckpts/qwen3_0_6B_base_olmo/model_and_optim}"
RUN_NAME="${RUN_NAME:-q06b-dense-contra-n20-sft-local}"
echo "=== HOST=$(hostname) NGPU=$NGPU MAX_STEPS=$MAX_STEPS BASE=$BASE_CKPT RUN=$RUN_NAME START=$(date '+%F %T') ==="

# Stage the (tiny) data + a FRESH work dir on node-local /data: shared-NFS work dirs accumulate
# stale NFSv4 locks from killed jobs (ranks then hang in nfs_wait_bit_killable before step 1).
LOCAL_ROOT=/data/prasann/contra_n20_smoke
DATA_LOCAL=$LOCAL_ROOT/data
mkdir -p "$DATA_LOCAL"
cp -u /scratch/users/prasann/longctx_sft_qwen/contradiction_n20/* "$DATA_LOCAL"/
WORK_LOCAL=$(fresh_workdir "$LOCAL_ROOT")
SAVE_LOCAL=$LOCAL_ROOT/ckpts/$RUN_NAME
mkdir -p "$SAVE_LOCAL"

cd "$REPO"
MAXARG=""; [ "$MAX_STEPS" -gt 0 ] && MAXARG="--max-steps $MAX_STEPS"
torchrun --nproc_per_node="$NGPU" --master_port="$MASTER_PORT" \
  src/scripts/train/memexpress/attn_explore/Qwen3-0.6B-dense-contradiction-n20-SFT-local.py \
  --run-name "$RUN_NAME" --base-checkpoint "$BASE_CKPT" \
  --data-path "$DATA_LOCAL" --work-dir "$WORK_LOCAL" --save-folder "$SAVE_LOCAL" --num-workers 0 \
  $MAXARG $WANDB_FLAG
echo "=== DONE $(date '+%F %T')  rc=$? ==="
