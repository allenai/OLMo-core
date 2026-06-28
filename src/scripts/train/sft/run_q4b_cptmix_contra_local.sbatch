#!/bin/bash
# Local (no-Beaker/weka) dense Qwen3-4B CPT-mix contradiction SFT on a jsteinhardt H200 node.
# Node + hyperparameters are env-parameterized so several CPT_FRAC/LR configs run in parallel
# across mooney / cubbins / sneetches.
#
#   sbatch --job-name=cptmix-f70 --nodelist=mooney \
#     --export=ALL,CPT_FRAC=0.70,LR=5e-5,RUN_NAME=q4b-cptmix-contra-f70-lr5e5 \
#     src/scripts/train/sft/run_q4b_cptmix_contra_local.sbatch
#
# Smoke (few steps, 2 GPUs):  add MAX_STEPS=8,NGPU=2 to --export.
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=mooney
#SBATCH --gres=gpu:H200:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --output=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/.cptmix_%x_%j.log

set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
NGPU="${NGPU:-8}"
MAX_STEPS="${MAX_STEPS:-150}"
CPT_FRAC="${CPT_FRAC:-0.70}"
LR="${LR:-5e-5}"
SEQ_LEN="${SEQ_LEN:-8192}"
RUN_NAME="${RUN_NAME:-q4b-cptmix-contra-f${CPT_FRAC}-lr${LR}}"
echo "=== HOST=$(hostname) NGPU=$NGPU MAX_STEPS=$MAX_STEPS CPT_FRAC=$CPT_FRAC LR=$LR SEQ_LEN=$SEQ_LEN RUN=$RUN_NAME START=$(date '+%F %T') ==="

source /system/linux/miniforge-3.13/etc/profile.d/conda.sh
# Prefer a node-local /data env clone (fast imports); fall back to the shared NFS env.
ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$ENV" ] || ENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
conda activate "$ENV"
echo "env=$ENV"

# Use THIS working tree's olmo_core (composable mixing + qwen3_4B), not the env's editable clone.
export PYTHONPATH="$REPO/src"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
# NCCL on these jsteinhardt nodes hangs on the first collective when it tries IB/P2P; force
# single-node-friendly transports. NCCL_DEBUG=WARN surfaces any remaining transport errors.
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"

# wandb: olmo-core's WandBCallback REQUIRES WANDB_API_KEY env var (does NOT read ~/.netrc).
if [ -z "${WANDB_API_KEY:-}" ]; then
  WANDB_API_KEY=$(awk '/machine api.wandb.ai/{f=1} f&&/password/{print $2; exit}' "$HOME/.netrc" 2>/dev/null)
  export WANDB_API_KEY
fi
WANDB_FLAG=""
[ -z "${WANDB_API_KEY:-}" ] && WANDB_FLAG="--no-wandb" && echo "WARN: no wandb creds -> --no-wandb"
[ -n "${NO_WANDB:-}" ] && WANDB_FLAG="--no-wandb" && echo "NO_WANDB set -> --no-wandb"

# Pin clean GPUs (jsteinhardt H200 nodes occasionally hand out a rogue-occupied GPU).
mapfile -t CLEAN < <(nvidia-smi --query-gpu=memory.used,uuid --format=csv,noheader,nounits | awk -F', ' '$1+0 < 2000 {print $2}')
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${CLEAN[*]:0:$NGPU}")
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  (clean=${#CLEAN[@]})"

cd "$REPO"
TARGET_TOKENS="${TARGET_TOKENS:-0}"
MASTER_PORT=$(( 20000 + RANDOM % 20000 ))
torchrun --nproc_per_node="$NGPU" --master_port="$MASTER_PORT" \
  src/scripts/train/sft/Qwen3-4B-dense-cptmix-contra-local.py \
  --run-name "$RUN_NAME" --cpt-frac "$CPT_FRAC" --lr "$LR" \
  --seq-len "$SEQ_LEN" --max-steps "$MAX_STEPS" --target-tokens "$TARGET_TOKENS" \
  --wandb-group q4b-cptmix-contra $WANDB_FLAG
echo "=== DONE $(date '+%F %T')  rc=$? ==="
