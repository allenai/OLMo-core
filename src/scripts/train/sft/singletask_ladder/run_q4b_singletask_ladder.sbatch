#!/bin/bash
# LOCAL single-task length-ladder SFT (dense or compressive) on ONE jsteinhardt H200 node.
# Trains Qwen3-4B on one task's whole 4k/8k/16k/32k ladder, NO CPT mix. Checkpoints -> node-local /data.
#
#   sbatch --job-name=dl-contra --export=ALL,VARIANT=dense,TASK=contra \
#     src/scripts/train/sft/singletask_ladder/run_q4b_singletask_ladder.sbatch
#
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive
#SBATCH --gres=gpu:H200:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=04:00:00
#SBATCH --output=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/.stl_%x_%j.log

set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
NGPU="${NGPU:-8}"
VARIANT="${VARIANT:-dense}"            # dense | compressive (compressive base must be local: pass BASE_CKPT)
TASK="${TASK:?set TASK=contra|nq|oolong|rerank|outlier}"
LR="${LR:-1e-5}"
EPOCHS="${EPOCHS:-2}"
MAX_STEPS="${MAX_STEPS:-0}"            # >0 = smoke (overrides epochs)
WANDB="${WANDB:-1}"
DATA_ROOT="${DATA_ROOT:-/scratch/users/prasann/single_task_ladders}"
BASE_CKPT="${BASE_CKPT:-}"
RUN_NAME="${RUN_NAME:-q4b-${VARIANT}-${TASK}-ladder32k-local}"
SAVE_FOLDER=/data/prasann/singletask_ladder_runs/$RUN_NAME
echo "=== HOST=$(hostname) RUN=$RUN_NAME VARIANT=$VARIANT TASK=$TASK EPOCHS=$EPOCHS NGPU=$NGPU START=$(date '+%F %T') ==="

source /system/linux/miniforge-3.13/etc/profile.d/conda.sh
OLMOENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$OLMOENV" ] || OLMOENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo

export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=lo
if [ "$WANDB" = "1" ]; then
  export WANDB_API_KEY=$(awk '/api.wandb.ai/{getline; if($1=="login")getline; for(i=1;i<=NF;i++) if($i=="password") print $(i+1)}' ~/.netrc 2>/dev/null | head -1)
  [ -z "${WANDB_API_KEY:-}" ] && export WANDB_API_KEY=$(grep -A2 'machine api.wandb.ai' ~/.netrc | awk '/password/{print $2}' | head -1)
fi

# Trust SLURM's GPU allocation (nvidia-smi is not cgroup-isolated here).
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  mapfile -t CLEAN < <(nvidia-smi --query-gpu=memory.used,uuid --format=csv,noheader,nounits | awk -F', ' '$1+0 < 2000 {print $2}')
  export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${CLEAN[*]:0:$NGPU}")
fi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

conda activate "$OLMOENV"
export PYTHONPATH="$REPO/src"
cd "$REPO"
DUR="--epochs $EPOCHS"; [ "$MAX_STEPS" -gt 0 ] && DUR="--max-steps $MAX_STEPS"
WBARG=""; [ "$WANDB" != "1" ] && WBARG="--no-wandb"
BASEARG=""; [ -n "$BASE_CKPT" ] && BASEARG="--base-ckpt $BASE_CKPT"
trc=1
for attempt in 1 2 3; do
  MASTER_PORT=$(( 20000 + RANDOM % 20000 ))
  echo "--- train attempt $attempt (port $MASTER_PORT) $(date '+%T') ---"
  torchrun --nproc_per_node="$NGPU" --master_port="$MASTER_PORT" \
    src/scripts/train/sft/singletask_ladder/Qwen3-4B-singletask-ladder-SFT-local.py \
    --variant "$VARIANT" --task "$TASK" --run-name "$RUN_NAME" --save-folder "$SAVE_FOLDER" \
    --data-root "$DATA_ROOT" --lr "$LR" $DUR $WBARG $BASEARG
  trc=$?; echo "train attempt $attempt rc=$trc"
  [ $trc -eq 0 ] && break
  echo "  (retrying after failure)"; sleep 10
done
[ $trc -ne 0 ] && { echo "TRAIN FAILED after retries"; exit $trc; }
STEP=$(ls -d "$SAVE_FOLDER"/step* 2>/dev/null | sed 's#.*/step##' | sort -n | tail -1)
echo "=== DONE $(date '+%T') latest=$SAVE_FOLDER/step${STEP} ==="