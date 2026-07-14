#!/bin/bash
# Local COMPRESSIVE-landmark Qwen3-0.6B contradiction-n20 (no-cot) SFT on mooney. Sibling of
# run_q06b_fast_contra_n20_local_mooney.sbatch. Inits from the compressive-pretrained 0.6B 64k base
# on mooney /data (the launcher's default) and uses THIS tree's olmo_core.
#
# Smoke:  sbatch --export=ALL,MAX_STEPS=10,NGPU=2 run_q06b_comp_contra_n20_local_mooney.sbatch
# Full:   sbatch run_q06b_comp_contra_n20_local_mooney.sbatch
#SBATCH --job-name=q06b-comp-contra-local
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=mooney
#SBATCH --gres=gpu:H200:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/.q06b_comp_contra_local.log

set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
NGPU="${NGPU:-8}"
MAX_STEPS="${MAX_STEPS:-0}"
MEM_FREQ="${MEM_FREQ:-63}"
RUN_NAME="${RUN_NAME:-q06b-comp-contra-n20-sft-local}"
echo "=== HOST=$(hostname) NGPU=$NGPU MAX_STEPS=$MAX_STEPS MEM_FREQ=$MEM_FREQ RUN=$RUN_NAME START=$(date '+%F %T') ==="

source /system/linux/miniforge-3.13/etc/profile.d/conda.sh
ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$ENV" ] || ENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
conda activate "$ENV"

export PYTHONPATH="$REPO/src"
export TOKENIZERS_PARALLELISM=false
if [ -z "${WANDB_API_KEY:-}" ]; then
  WANDB_API_KEY=$(awk '/machine api.wandb.ai/{f=1} f&&/password/{print $2; exit}' "$HOME/.netrc" 2>/dev/null)
  export WANDB_API_KEY
fi
WANDB_FLAG=""
[ -z "${WANDB_API_KEY:-}" ] && WANDB_FLAG="--no-wandb" && echo "WARN: no wandb creds -> --no-wandb"

mapfile -t CLEAN < <(nvidia-smi --query-gpu=memory.used,uuid --format=csv,noheader,nounits | awk -F', ' '$1+0 < 2000 {print $2}')
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${CLEAN[*]:0:$NGPU}")
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

cd "$REPO"
MAXARG=""; [ "$MAX_STEPS" -gt 0 ] && MAXARG="--max-steps $MAX_STEPS"
torchrun --nproc_per_node="$NGPU" \
  src/scripts/train/attn_explore/Qwen3-0.6B-compressive-contradiction-n20-SFT-local.py \
  --run-name "$RUN_NAME" --mem-freq "$MEM_FREQ" $MAXARG $WANDB_FLAG
echo "=== DONE $(date '+%F %T')  rc=$? ==="
