#!/bin/bash
# SHORT fast-landmark CPT (~40M tokens) of Qwen3-4B from the RAW base, to test whether a tiny
# landmark continued-pretrain is enough for landmark attention to then learn the contradiction task
# (vs the full 10B run / saved step1000=4.2B). CPT-only: cpt-frac 1.0, all SFT/task fracs 0, so the
# only data source is the local dolma3_longmino 40M-token slice (CPT_DATA_ROOT in the launcher).
# No eval here -- a separate SFT+eval sbatch consumes the saved checkpoint.
#
#   cd $REPO; sbatch src/scripts/train/sft/run_q4b_fastlm_cpt40m.sbatch
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive
#SBATCH --gres=gpu:H200:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=06:00:00
#SBATCH --output=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/.cpt40m_%x_%j.log

set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
NGPU="${NGPU:-8}"
MODEL="${MODEL:-qwen3_4B}"
LR="${LR:-3.2e-4}"
SEQ_LEN="${SEQ_LEN:-8192}"
MEM_FREQ="${MEM_FREQ:-63}"
TARGET_TOKENS="${TARGET_TOKENS:-40000000}"
RUN_NAME="${RUN_NAME:-q4b-fastlm-cpt40m}"
# RAW (no-landmark-CPT) Qwen3-4B base, model-only distcp transferred from weka amandab/Qwen3-4B.
BASE_CKPT="${BASE_CKPT:-/scratch/users/prasann/cpt_mix_ckpts/q4b-base-modelonly/model_and_optim}"
SAVE_FOLDER="${SAVE_FOLDER:-/data/prasann/cpt_mix_runs/$RUN_NAME}"
echo "=== HOST=$(hostname) RUN=$RUN_NAME SEQ=$SEQ_LEN LR=$LR TARGET_TOKENS=$TARGET_TOKENS BASE=$BASE_CKPT START=$(date '+%F %T') ==="

[ -f "$BASE_CKPT/.metadata" ] || { echo "BASE_CKPT missing (.metadata not found): $BASE_CKPT"; exit 2; }

source /system/linux/miniforge-3.13/etc/profile.d/conda.sh
OLMOENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$OLMOENV" ] || OLMOENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo

export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=lo

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
  echo "--- CPT attempt $attempt (port $MASTER_PORT) $(date '+%T') ---"
  torchrun --nproc_per_node="$NGPU" --master_port="$MASTER_PORT" \
    src/scripts/train/sft/Qwen3-4B-fast-landmark-cptmix-5task-local.py \
    --run-name "$RUN_NAME" --save-folder "$SAVE_FOLDER" --model "$MODEL" \
    --base-ckpt "$BASE_CKPT" \
    --cpt-frac 1.0 --tulu-frac 0 --nq-frac 0 --oolong-frac 0 --rerank-frac 0 --outlier-frac 0 \
    --lr "$LR" --seq-len "$SEQ_LEN" --mem-freq "$MEM_FREQ" --target-tokens "$TARGET_TOKENS" --no-wandb
  trc=$?; echo "CPT attempt $attempt rc=$trc"
  [ $trc -eq 0 ] && break
  echo "  (retrying after failure)"; sleep 10
done
[ $trc -ne 0 ] && { echo "CPT FAILED after retries"; exit $trc; }

STEP_N=$(ls -d "$SAVE_FOLDER"/step* 2>/dev/null | sed 's#.*/step##' | sort -n | tail -1)
STEP_DIR="$SAVE_FOLDER/step${STEP_N}"
echo "latest step dir: $STEP_DIR"
[ -f "$STEP_DIR/model_and_optim/.metadata" ] || { echo "no final checkpoint"; exit 2; }
[ -f "$STEP_DIR/config.json" ] || echo "WARN: config.json missing in $STEP_DIR"

# /data is node-local on these GPU nodes; copy the final ckpt to shared /scratch so a later
# SFT job (possibly on a different node) can load it as BASE_CKPT.
SHARED_DST="/scratch/users/prasann/cpt_mix_ckpts/$RUN_NAME/step${STEP_N}"
echo "=== copying ckpt to shared NFS: $SHARED_DST ==="
mkdir -p "$SHARED_DST"
cp -r "$STEP_DIR/." "$SHARED_DST/"
[ -f "$SHARED_DST/model_and_optim/.metadata" ] || { echo "shared copy failed"; exit 2; }
echo "=== CPT DONE $(date '+%T') node_ckpt=$STEP_DIR shared_ckpt=$SHARED_DST step=$STEP_N ==="
