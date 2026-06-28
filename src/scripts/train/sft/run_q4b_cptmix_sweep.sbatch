#!/bin/bash
# Self-contained CPT-mix sweep job for ONE config on ONE jsteinhardt H200 node:
#   1) train dense Qwen3-4B (contradiction + CPT mix) -> checkpoint on node-local /data
#   2) export the final step -> HF (node-local)
#   3) eval RULER (2 smallest ctx) + contradiction (size 100) -> summary JSON on shared NFS
# Keeping everything node-local avoids slow/flaky 50GB distcp writes to NFS and cross-node reads.
#
#   sbatch --job-name=sw-f70-lr5e5 --nodelist=sneetches --qos=preemptive --gres=gpu:H200:8 \
#     --export=ALL,NGPU=8,CPT_FRAC=0.70,LR=5e-5,TARGET_TOKENS=48000000,RUN_NAME=q4b-cptmix-f70-lr5e5 \
#     src/scripts/train/sft/run_q4b_cptmix_sweep.sbatch
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive
#SBATCH --nodelist=sneetches
#SBATCH --gres=gpu:H200:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=06:00:00
#SBATCH --output=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/.sweep_%x_%j.log

set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
CR=/scratch/users/prasann/corpus-reasoning
NGPU="${NGPU:-8}"
MODEL="${MODEL:-qwen3_4B}"                    # qwen3_4B or qwen3_0_6B (fast iteration)
CPT_FRAC="${CPT_FRAC:-0.70}"
TULU_FRAC="${TULU_FRAC:-0.0}"                 # diverse chat SFT fraction (contradiction = 1-cpt-tulu)
NQ_FRAC="${NQ_FRAC:-0.0}"                       # fixed (aligned) NQ retrieval fraction
OOLONG_FRAC="${OOLONG_FRAC:-0.0}"; RERANK_FRAC="${RERANK_FRAC:-0.0}"; OUTLIER_FRAC="${OUTLIER_FRAC:-0.0}"
LR="${LR:-5e-5}"
SEQ_LEN="${SEQ_LEN:-8192}"
TARGET_TOKENS="${TARGET_TOKENS:-48000000}"   # ~1 mixture epoch (47.9M tok) by default
BATCH_TOKENS="${BATCH_TOKENS:-0}"   # 0 = seq_len*world_size; set e.g. 1048576 for CPT-scale batch
MAX_STEPS="${MAX_STEPS:-0}"
RUN_NAME="${RUN_NAME:-q4b-cptmix-f${CPT_FRAC}-lr${LR}}"
# Export skeleton base-model must match the trained model size.
if [ "$MODEL" = "qwen3_0_6B" ]; then BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-0.6B-Base}"; else BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B-Base}"; fi
SAVE_FOLDER=/data/prasann/cpt_mix_runs/$RUN_NAME
HF_OUT=/data/prasann/cpt_mix_hf/${RUN_NAME}_hf
RESULT=$CR/outputs/eval_results/${RUN_NAME}_lc.json
echo "=== HOST=$(hostname) RUN=$RUN_NAME CPT_FRAC=$CPT_FRAC LR=$LR TARGET_TOKENS=$TARGET_TOKENS NGPU=$NGPU START=$(date '+%F %T') ==="

source /system/linux/miniforge-3.13/etc/profile.d/conda.sh
OLMOENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$OLMOENV" ] || OLMOENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
EVALENV=/scratch/users/prasann/conda/envs/corpus-reasoning-eval

export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=lo

# Trust SLURM's GPU allocation (it sets CUDA_VISIBLE_DEVICES per job). Do NOT override via a
# node-wide nvidia-smi scan: nvidia-smi is not cgroup-isolated here, so a node-wide scan picks
# GPUs allocated to OTHER co-located jobs and collides. Only fall back to scanning if SLURM gave
# us nothing.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  mapfile -t CLEAN < <(nvidia-smi --query-gpu=memory.used,uuid --format=csv,noheader,nounits | awk -F', ' '$1+0 < 2000 {print $2}')
  export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${CLEAN[*]:0:$NGPU}")
fi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# ---------- 1) TRAIN ----------
conda activate "$OLMOENV"
export PYTHONPATH="$REPO/src"
cd "$REPO"
MAXARG="--target-tokens $TARGET_TOKENS"; [ "$MAX_STEPS" -gt 0 ] && MAXARG="--max-steps $MAX_STEPS --target-tokens 0"
echo "=== [1/3] TRAIN $(date '+%T') ==="
# Retry on the intermittent native SIGSEGV (restarts training; save_overwrite handles the ckpt dir).
trc=1
for attempt in 1 2 3; do
  MASTER_PORT=$(( 20000 + RANDOM % 20000 ))
  echo "--- train attempt $attempt (port $MASTER_PORT) $(date '+%T') ---"
  torchrun --nproc_per_node="$NGPU" --master_port="$MASTER_PORT" \
    src/scripts/train/sft/Qwen3-4B-dense-cptmix-contra-local.py \
    --run-name "$RUN_NAME" --save-folder "$SAVE_FOLDER" --model "$MODEL" \
    --cpt-frac "$CPT_FRAC" --tulu-frac "$TULU_FRAC" --nq-frac "$NQ_FRAC" --oolong-frac "$OOLONG_FRAC" --rerank-frac "$RERANK_FRAC" --outlier-frac "$OUTLIER_FRAC" --lr "$LR" --seq-len "$SEQ_LEN" --batch-tokens "$BATCH_TOKENS" $MAXARG --no-wandb
  trc=$?; echo "train attempt $attempt rc=$trc"
  [ $trc -eq 0 ] && break
  echo "  (retrying after failure)"; sleep 10
done
[ $trc -ne 0 ] && { echo "TRAIN FAILED after retries"; exit $trc; }

STEP_DIR=$(ls -d "$SAVE_FOLDER"/step* 2>/dev/null | sed 's#.*/step##' | sort -n | tail -1)
STEP_DIR="$SAVE_FOLDER/step${STEP_DIR}"
echo "latest step dir: $STEP_DIR"
[ -f "$STEP_DIR/model_and_optim/.metadata" ] || { echo "no final checkpoint"; exit 2; }

# ---------- 2) EXPORT -> HF ----------
echo "=== [2/3] EXPORT $(date '+%T') ==="
export PYTHONPATH="$CR"
cd "$CR"
python -u scripts/train/export_olmo_to_hf.py --save-folder "$STEP_DIR" --ckpt "$STEP_DIR" \
  --hf-out "$HF_OUT" --base-model "$BASE_MODEL"
erc=$?; echo "export rc=$erc"; [ $erc -ne 0 ] && { echo "EXPORT FAILED"; exit $erc; }
conda deactivate

# ---------- 3) EVAL ----------
echo "=== [3/3] EVAL $(date '+%T') ==="
conda activate "$EVALENV"
export PYTHONPATH="$CR"
python -u scripts/eval/eval_lc_fast.py \
  --base-model "$HF_OUT" --out "$RESULT" \
  --max-test-samples "${MAX_SAMPLES:-100}" --ruler-lengths "${RULER_LENGTHS:-L1024,L2048}" --max-model-len 16384
echo "=== DONE $(date '+%T') result=$RESULT ==="