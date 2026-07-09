#!/bin/bash
# Attention-pattern EXPLORATION (contradiction, Qwen3-0.6B, LOCAL mooney). One sbatch, three VARIANTs,
# all init from the SAME shared dense CPT base for a fair comparison of the attention MASK only:
#   VARIANT=full   -> full (dense) attention on the PLAIN contra shard (baseline the others must match)
#   VARIANT=chunked-> DocumentChunkedAttention cross_doc_mode="chunked"  (the known-to-LOSE control)
#   VARIANT=hier   -> DocumentChunkedAttention cross_doc_mode="hierarchical_dilated" (DIL_N/DIL_M)
#                     -- a strict SUPERSET of chunked with whole-document dilation + full-context free
#                        query/answer. NOT the positional dilated_sliding_window.
#
# NFS-safe (see local-train-nfs-log-hang memory): --output on node-local /data, direct PATH (no
# conda activate), PYTHONWARNINGS=ignore, data+base+workdir+ckpt all staged to node-local /data.
#
#   sbatch --export=ALL,VARIANT=full,NDOCS=100,RUN=q06b-full-c100 run_q06b_attn_explore_mooney.sbatch
#   sbatch --export=ALL,VARIANT=chunked,NDOCS=100,RUN=q06b-chunked-c100 run_q06b_attn_explore_mooney.sbatch
#   sbatch --export=ALL,VARIANT=hier,DIL_N=3,DIL_M=2,NDOCS=100,RUN=q06b-hier-n3m2-c100 run_q06b_attn_explore_mooney.sbatch
#SBATCH --job-name=q06b-attn-explore
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=mooney
#SBATCH --gres=gpu:H200:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=/data/prasann/joblogs/q06b_attn_explore_%x_%j.log

set -uo pipefail
# Mirror this job's node-local /data log to shared /scratch on exit so it is readable off-node.
MIRROR_DIR=/scratch/users/prasann/attn_explore_logs
mkdir -p "$MIRROR_DIR" 2>/dev/null || true
_SRC_LOG="/data/prasann/joblogs/q06b_attn_explore_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
_DST_LOG="$MIRROR_DIR/train_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
_mirror_log() { cp -f "$_SRC_LOG" "$_DST_LOG" 2>/dev/null || true; }
# Periodic mirror (rank-0 slurm log -> shared scratch) so it is readable off-node while running.
( while true; do _mirror_log; sleep 30; done ) &
_MIRROR_PID=$!
trap '_mirror_log; kill $_MIRROR_PID 2>/dev/null' EXIT
REPO="${REPO:-/accounts/projects/berkeleynlp/prasann/projects/OLMo-core}"
NGPU="${NGPU:-8}"
MAX_STEPS="${MAX_STEPS:-0}"
VARIANT="${VARIANT:?set VARIANT=full|chunked|hier}"
NDOCS="${NDOCS:-100}"
SEQ_LEN="${SEQ_LEN:-6144}"
EPOCHS="${EPOCHS:-3}"
DIL_N="${DIL_N:-3}"
DIL_M="${DIL_M:-2}"
RUN="${RUN:-q06b-${VARIANT}-c${NDOCS}}"
# Shared dense CPT base (on /scratch) -> stage to node-local /data once; all variants share it.
BASE_SRC="${BASE_SRC:-/scratch/users/prasann/cpt_mix_ckpts/q06b-dense-cpt-modelonly}"
echo "=== HOST=$(hostname) VARIANT=$VARIANT NDOCS=$NDOCS SEQ_LEN=$SEQ_LEN DIL_N=$DIL_N DIL_M=$DIL_M RUN=$RUN START=$(date '+%F %T') ==="

ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$ENV" ] || ENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$REPO/src"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONWARNINGS=ignore
export PYTHONUNBUFFERED=1  # flush CE/loss lines live so the /scratch mirror is readable during the run
if [ -z "${WANDB_API_KEY:-}" ]; then
  WANDB_API_KEY=$(awk '/machine api.wandb.ai/{f=1} f&&/password/{print $2; exit}' "$HOME/.netrc" 2>/dev/null)
  export WANDB_API_KEY
fi
WANDB_FLAG=""
[ -z "${WANDB_API_KEY:-}" ] && WANDB_FLAG="--no-wandb" && echo "WARN: no wandb creds -> --no-wandb"

mapfile -t CLEAN < <(nvidia-smi --query-gpu=memory.used,uuid --format=csv,noheader,nounits | awk -F', ' '$1+0 < 2000 {print $2}')
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${CLEAN[*]:0:$NGPU}")
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# ---- base: read the shared dense CPT distcp directly from /scratch (distcp load = reads, NFS-safe;
# staging to /data via cp dropped the hidden .metadata -> silent train-from-scratch, CE~12). ----
ROOT=/data/prasann/attn_explore
BASE_CKPT="$BASE_SRC/model_and_optim"
if [ ! -f "$BASE_CKPT/.metadata" ]; then
  echo "FATAL: base checkpoint $BASE_CKPT missing .metadata -> would silently train from scratch"; exit 3
fi

if [ "$VARIANT" = "full" ]; then
  DATA_SRC=/scratch/users/prasann/longctx_sft_qwen/contradiction_n${NDOCS}_nocot
else
  DATA_SRC=/scratch/users/prasann/longctx_sft_qwen/contradiction_n${NDOCS}_docdense_nocot
fi
DATA_LOCAL=$ROOT/data/$(basename "$DATA_SRC")
mkdir -p "$DATA_LOCAL"
cp -un "$DATA_SRC"/* "$DATA_LOCAL"/ 2>/dev/null || cp -n "$DATA_SRC"/* "$DATA_LOCAL"/ 2>/dev/null || true
WORK_LOCAL=$ROOT/cache-$SLURM_JOB_ID
mkdir -p "$WORK_LOCAL"
SAVE_LOCAL=$ROOT/ckpts/$RUN
mkdir -p "$SAVE_LOCAL"
echo "BASE_CKPT=$BASE_CKPT DATA_LOCAL=$DATA_LOCAL SAVE=$SAVE_LOCAL"

cd "$REPO"
MAXARG=""; [ "$MAX_STEPS" -gt 0 ] && MAXARG="--max-steps $MAX_STEPS"
MASTER_PORT=$((29000 + RANDOM % 1000))

if [ "$VARIANT" = "full" ]; then
  torchrun --nproc_per_node="$NGPU" --master_port="$MASTER_PORT" \
    src/scripts/train/sft/Qwen3-0.6B-dense-contradiction-n20-SFT-local.py \
    --run-name "$RUN" --base-checkpoint "$BASE_CKPT" \
    --data-path "$DATA_LOCAL" --work-dir "$WORK_LOCAL" --save-folder "$SAVE_LOCAL" \
    --seq-len "$SEQ_LEN" --epochs "$EPOCHS" --num-workers 0 --save-checkpoint \
    $MAXARG $WANDB_FLAG
else
  CDM="chunked"; DILARGS=""
  if [ "$VARIANT" = "hier" ]; then CDM="hierarchical_dilated"; DILARGS="--dilation-n $DIL_N --dilation-m $DIL_M"; fi
  if [ "$VARIANT" = "random" ]; then CDM="random_doc"; DILARGS="--doc-keep-prob ${KEEP:-0.5}"; fi
  # docchunk launcher has no wandb wiring; it always runs offline-safe.
  torchrun --nproc_per_node="$NGPU" --master_port="$MASTER_PORT" \
    src/scripts/train/sft/Qwen3-0.6B-docchunk-mask-mix-contradiction-SFT-local.py \
    --run-name "$RUN" --base-checkpoint "$BASE_CKPT" \
    --data-dir "$DATA_LOCAL" --work-dir "$WORK_LOCAL" --save-folder "$SAVE_LOCAL" \
    --seq-len "$SEQ_LEN" --epochs "$EPOCHS" --grad-accum 8 --mix-mode none \
    --cross-doc-mode "$CDM" $DILARGS --save-checkpoint --no-compile \
    $MAXARG
fi
echo "=== DONE $(date '+%F %T')  rc=$? ==="
