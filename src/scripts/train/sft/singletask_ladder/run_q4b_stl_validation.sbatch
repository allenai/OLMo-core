#!/bin/bash
# Single-task-ladder VALIDATION run (dense) at a capped context (default 14k) on one jsteinhardt
# H200 node. Bounded by MAX_STEPS so it is a quick sanity/early-signal run, not a full epoch.
# Data: /data/prasann/stl_20k/<task> (20k examples/task, continuous lengths <=40960).
#
#   sbatch --job-name=val-contra --export=ALL,TASK=contra,SEQ_LEN=14336,MAX_STEPS=1500 \
#     src/scripts/train/sft/singletask_ladder/run_q4b_stl_validation.sbatch
#
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive
#SBATCH -w mooney
#SBATCH --gres=gpu:H200:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=03:00:00
#SBATCH --output=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/.stlval_%x_%j.log
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
NGPU="${NGPU:-8}"
VARIANT="${VARIANT:-dense}"                  # dense only locally (compressive base is weka -> Beaker)
TASK="${TASK:-contra}"                        # contra | nq | oolong | rerank | outlier
SEQ_LEN="${SEQ_LEN:-14336}"                   # 14k cap; multiple of 64 (compressive constraint)
MAX_STEPS="${MAX_STEPS:-1500}"                # bounded validation (>0 overrides epochs)
BATCH_TOKENS="${BATCH_TOKENS:-0}"             # 0 = 1 window/step; >seq_len = multi-window (avoids 0-loss-token NaN)
SAVE_INTERVAL="${SAVE_INTERVAL:-400}"         # checkpoint every N steps (lower = more resilient to node blips)
LR="${LR:-1e-5}"
DATA_ROOT="${DATA_ROOT:-/data/prasann/stl_20k}"
RUN_NAME="${RUN_NAME:-q4b-${VARIANT}-${TASK}-${SEQ_LEN}val}"
SAVE_FOLDER=/data/prasann/singletask_ladder_runs/$RUN_NAME
WBGROUP="${WBGROUP:-q4b-stl-14kval}"
echo "=== HOST=$(hostname) RUN=$RUN_NAME VARIANT=$VARIANT TASK=$TASK SEQ=$SEQ_LEN STEPS=$MAX_STEPS DATA=$DATA_ROOT START=$(date '+%F %T') ==="

source /system/linux/miniforge-3.13/etc/profile.d/conda.sh
OLMOENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$OLMOENV" ] || OLMOENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo

export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=lo
export WANDB_API_KEY=$(awk '/api.wandb.ai/{getline; if($1=="login")getline; for(i=1;i<=NF;i++) if($i=="password") print $(i+1)}' ~/.netrc 2>/dev/null | head -1)
[ -z "${WANDB_API_KEY:-}" ] && export WANDB_API_KEY=$(grep -A2 'machine api.wandb.ai' ~/.netrc | awk '/password/{print $2}' | head -1)

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  mapfile -t CLEAN < <(nvidia-smi --query-gpu=memory.used,uuid --format=csv,noheader,nounits | awk -F', ' '$1+0 < 2000 {print $2}')
  export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${CLEAN[*]:0:$NGPU}")
fi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

conda activate "$OLMOENV"
export PYTHONPATH="$REPO/src"
cd "$REPO"
trc=1
for attempt in 1 2 3; do
  MASTER_PORT=$(( 20000 + RANDOM % 20000 ))
  echo "--- train attempt $attempt (port $MASTER_PORT) $(date '+%T') ---"
  torchrun --nproc_per_node="$NGPU" --master_port="$MASTER_PORT" \
    src/scripts/train/sft/singletask_ladder/Qwen3-4B-singletask-ladder-SFT-local.py \
    --variant "$VARIANT" --task "$TASK" --data-root "$DATA_ROOT" \
    --seq-len "$SEQ_LEN" --max-steps "$MAX_STEPS" --lr "$LR" --batch-tokens "$BATCH_TOKENS" --save-interval "$SAVE_INTERVAL" \
    --run-name "$RUN_NAME" --save-folder "$SAVE_FOLDER" --wandb-group "$WBGROUP"
  trc=$?; echo "train attempt $attempt rc=$trc"
  [ $trc -eq 0 ] && break
  echo "  (retrying after failure)"; sleep 10
done
echo "=== DONE rc=$trc $(date '+%F %T') ==="
