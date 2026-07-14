#!/bin/bash
# Attention-pattern EXPLORATION (contradiction, hybrid Qwen3.5-0.8B, LOCAL). One sbatch, three
# docchunk VARIANTs, all init from the SAME Qwen3.5-0.8B-Base olmo distcp for a fair comparison of
# the attention MASK only (applied to the 6 full-attention blocks; the 18 Gated DeltaNet blocks are
# linear attention and stay unmasked):
#   VARIANT=chunked -> DocumentChunkedAttention cross_doc_mode="chunked"
#   VARIANT=hier    -> DocumentChunkedAttention cross_doc_mode="hierarchical_dilated" (DIL_N/DIL_M/DIL_CYCLE)
#   VARIANT=random  -> DocumentChunkedAttention cross_doc_mode="random_doc" (KEEP)
#
# Data: Qwen3.5-TOKENIZED docdense shard (box 248049/248050, eos 248044) -- NOT the Qwen3 shard.
# NFS-safe (see local-train-nfs-log-hang memory): --output on node-local /data, direct PATH (no
# conda activate), PYTHONWARNINGS=ignore, data+base+workdir+ckpt all staged to node-local /data.
#
#   sbatch --export=ALL,VARIANT=chunked,RUN=q35-08b-chunked-mixcurr-c20 --nodelist=mooney run_q35_08b_attn_explore.sbatch
#   sbatch --export=ALL,VARIANT=hier,DIL_N=4,DIL_M=2,DIL_CYCLE=3,RUN=q35-08b-hier-mixcurr-c20 --nodelist=cubbins run_q35_08b_attn_explore.sbatch
#   sbatch --export=ALL,VARIANT=random,KEEP=0.5,RUN=q35-08b-random-mixcurr-c20 --nodelist=mooney run_q35_08b_attn_explore.sbatch
#SBATCH --job-name=q35-attn-explore
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=/data/prasann/joblogs/q35_attn_explore_%x_%j.log

set -uo pipefail
# Mirror this job's node-local /data log to shared /scratch on exit so it is readable off-node.
MIRROR_DIR=/scratch/users/prasann/attn_explore_logs
mkdir -p "$MIRROR_DIR" 2>/dev/null || true
_SRC_LOG="/data/prasann/joblogs/q35_attn_explore_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
_DST_LOG="$MIRROR_DIR/train_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
_mirror_log() { cp -f "$_SRC_LOG" "$_DST_LOG" 2>/dev/null || true; }
( while true; do _mirror_log; sleep 30; done ) &
_MIRROR_PID=$!
trap '_mirror_log; kill $_MIRROR_PID 2>/dev/null' EXIT
REPO="${REPO:-/accounts/projects/berkeleynlp/prasann/projects/OLMo-core}"
NGPU="${NGPU:-8}"
MAX_STEPS="${MAX_STEPS:-0}"
VARIANT="${VARIANT:?set VARIANT=chunked|hier|random}"
SEQ_LEN="${SEQ_LEN:-2048}"
EPOCHS="${EPOCHS:-3}"
DIL_N="${DIL_N:-4}"
DIL_M="${DIL_M:-2}"
RUN="${RUN:-q35-08b-${VARIANT}-mixcurr-c20}"
# Qwen3.5-0.8B-Base olmo distcp (converted from HF, model-only) on shared /scratch; all variants share it.
BASE_SRC="${BASE_SRC:-/scratch/users/prasann/cpt_mix_ckpts/q35-08b-base-modelonly}"
DATA_SRC="${DATA_SRC:-/scratch/users/prasann/longctx_sft_qwen/contradiction_n20_docdense_nocot_qwen35}"
echo "=== HOST=$(hostname) VARIANT=$VARIANT SEQ_LEN=$SEQ_LEN DIL_N=$DIL_N DIL_M=$DIL_M RUN=$RUN START=$(date '+%F %T') ==="

ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$ENV" ] || ENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$REPO/src"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONWARNINGS=ignore
export PYTHONUNBUFFERED=1

mapfile -t CLEAN < <(nvidia-smi --query-gpu=memory.used,uuid --format=csv,noheader,nounits | awk -F', ' '$1+0 < 2000 {print $2}')
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${CLEAN[*]:0:$NGPU}")
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# ---- base: read the shared distcp directly from /scratch (distcp load = reads, NFS-safe; staging
# via cp can drop the hidden .metadata -> silent train-from-scratch). ----
ROOT=/data/prasann/attn_explore
BASE_CKPT="$BASE_SRC/model_and_optim"
if [ ! -f "$BASE_CKPT/.metadata" ]; then
  echo "FATAL: base checkpoint $BASE_CKPT missing .metadata -> would silently train from scratch"; exit 3
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

CDM="chunked"; DILARGS=""
if [ "$VARIANT" = "hier" ]; then
  CDM="hierarchical_dilated"; DILARGS="--dilation-n $DIL_N --dilation-m $DIL_M"
  [ -n "${DIL_CYCLE:-}" ] && DILARGS="$DILARGS --dilation-cycle $DIL_CYCLE"
fi
if [ "$VARIANT" = "random" ]; then CDM="random_doc"; DILARGS="--doc-keep-prob ${KEEP:-0.5}"; fi
# mask-mixing DEFAULT curriculum (anneal MIX_START_P -> MIX_END_P full-causal fraction); MIX_MODE=none to opt out.
MIXMODE="${MIX_MODE:-curriculum}"; MIXARGS="--mix-mode $MIXMODE"
[ "$MIXMODE" = "static" ]     && MIXARGS="$MIXARGS --standard-mix-prob ${MIX_PROB:-0.10}"
[ "$MIXMODE" = "curriculum" ] && MIXARGS="$MIXARGS --mix-start-p ${MIX_START_P:-0.8} --mix-end-p ${MIX_END_P:-0.0}"
FAARG=""; [ -n "${FULL_ATTN:-}" ] && FAARG="--full-attention-layers ${FULL_ATTN}"

torchrun --nproc_per_node="$NGPU" --master_port="$MASTER_PORT" \
  src/scripts/train/memexpress/attn_explore/Qwen3.5-0.8B-docchunk-mask-mix-contradiction-SFT-local.py \
  --run-name "$RUN" --base-checkpoint "$BASE_CKPT" \
  --data-dir "$DATA_LOCAL" --work-dir "$WORK_LOCAL" --save-folder "$SAVE_LOCAL" \
  --seq-len "$SEQ_LEN" --epochs "$EPOCHS" --grad-accum 8 $MIXARGS \
  --cross-doc-mode "$CDM" $DILARGS $FAARG --save-checkpoint --no-compile \
  $MAXARG
echo "=== DONE $(date '+%F %T')  rc=$? ==="
