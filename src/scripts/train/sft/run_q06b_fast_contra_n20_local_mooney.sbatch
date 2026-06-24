#!/bin/bash
# Local (no-Beaker) training of the Qwen3-0.6B fast-landmark contradiction-n20 SFT on an idle
# jsteinhardt H200 node (mooney). Uses THIS working tree's olmo_core (fast_landmark + composable
# landmark sources) via PYTHONPATH, the locally-regenerated shards on /scratch, and the locally
# converted Qwen3-0.6B base distcp on mooney's /data.
#
# Smoke test (a few steps):  sbatch --export=ALL,MAX_STEPS=10,NGPU=2 run_q06b_..._mooney.sbatch
# Full run (8 GPUs):         sbatch run_q06b_..._mooney.sbatch
#SBATCH --job-name=q06b-contra-local
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=mooney
#SBATCH --gres=gpu:H200:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/.q06b_contra_local.log

set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
NGPU="${NGPU:-8}"
MAX_STEPS="${MAX_STEPS:-0}"
echo "=== HOST=$(hostname)  NGPU=$NGPU  MAX_STEPS=$MAX_STEPS  START=$(date '+%F %T') ==="

source /system/linux/miniforge-3.13/etc/profile.d/conda.sh
ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$ENV" ] || ENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
conda activate "$ENV"

# Use THIS working tree's olmo_core (fast_landmark lives here), not the env's editable clone.
export PYTHONPATH="$REPO/src"
export TOKENIZERS_PARALLELISM=false
# wandb auth (set WANDB_API_KEY in your env/secrets; pass --no-wandb to skip).
WANDB_FLAG=""; [ -z "${WANDB_API_KEY:-}" ] && WANDB_FLAG="--no-wandb" && echo "WARN: no WANDB_API_KEY -> --no-wandb"

# Pin clean GPUs (defensive; mooney is usually idle).
mapfile -t CLEAN < <(nvidia-smi --query-gpu=memory.used,uuid --format=csv,noheader,nounits | awk -F', ' '$1+0 < 2000 {print $2}')
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${CLEAN[*]:0:$NGPU}")
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

cd "$REPO"
MAXARG=""; [ "$MAX_STEPS" -gt 0 ] && MAXARG="--max-steps $MAX_STEPS"
torchrun --nproc_per_node="$NGPU" \
  src/scripts/train/sft/Qwen3-0.6B-fast-landmark-contradiction-n20-SFT-local.py \
  --run-name q06b-fast-contra-n20-sft-local $MAXARG $WANDB_FLAG
echo "=== DONE $(date '+%F %T')  rc=$? ==="
