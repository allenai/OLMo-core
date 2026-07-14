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
VARIANT="${VARIANT:?set VARIANT=full|std|chunked|hier|random}"
# VALIDATE. The dispatch below is a two-way if/else (full -> dense script; EVERYTHING ELSE ->
# docchunk script), so an unrecognized VARIANT used to fall silently into the `else` and train
# **chunked** while every log line looked healthy -- a wrong experiment that reports a plausible
# number. Fail loudly instead.
case "$VARIANT" in
  full|std|chunked|hier|random) ;;
  *) echo "FATAL: unknown VARIANT='$VARIANT' (expected: full|std|chunked|hier|random)"; exit 2 ;;
esac
NDOCS="${NDOCS:-100}"
MODEL_SIZE="${MODEL_SIZE:-0.6B}"
SEQ_LEN="${SEQ_LEN:-6144}"
EPOCHS="${EPOCHS:-3}"
DIL_N="${DIL_N:-3}"
DIL_M="${DIL_M:-2}"
RUN="${RUN:-q06b-${VARIANT}-c${NDOCS}}"
# Shared dense CPT base (on /scratch) -> stage to node-local /data once; all variants share it.
BASE_SRC="${BASE_SRC:-/scratch/users/prasann/cpt_mix_ckpts/q06b-dense-cpt-modelonly}"
echo "=== HOST=$(hostname) MODEL_SIZE=$MODEL_SIZE VARIANT=$VARIANT NDOCS=$NDOCS SEQ_LEN=$SEQ_LEN DIL_N=$DIL_N DIL_M=$DIL_M RUN=$RUN START=$(date '+%F %T') ==="

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
# Cross the data/attention arms (dense script on the docdense shard, or vice-versa) to isolate whether
# a result comes from the shard or from the attention module.
DATA_SRC="${DATA_SRC_OVERRIDE:-$DATA_SRC}"
DATA_LOCAL=$ROOT/data/$(basename "$DATA_SRC")
mkdir -p "$DATA_LOCAL"
# rsync, NOT `cp -un`. `cp -un` is no-clobber: if a shard of the SAME NAME was staged here by an
# earlier job, a REBUILT shard is silently ignored and the job trains on the stale node-local copy --
# a wrong experiment that reports a plausible number. (This bit us: a shard rebuilt from 2000
# examples was ignored in favour of a stale 500-example copy.) rsync re-copies whenever size/mtime
# differ, so the node-local data always matches the source.
rsync -a --delete "$DATA_SRC/" "$DATA_LOCAL/"
# Fail loudly if the staged shard still disagrees with the source.
_src_n=$(python3 -c "import json;print(json.load(open('$DATA_SRC/metadata.json'))['num_instances'])" 2>/dev/null || echo x)
_dst_n=$(python3 -c "import json;print(json.load(open('$DATA_LOCAL/metadata.json'))['num_instances'])" 2>/dev/null || echo y)
if [ "$_src_n" != "$_dst_n" ]; then
  echo "FATAL: staged shard mismatch (src num_instances=$_src_n, node-local=$_dst_n)"; exit 4
fi
echo "DATA staged: num_instances=$_src_n from $DATA_SRC"
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
    src/scripts/train/attn_explore/Qwen3-0.6B-dense-contradiction-n20-SFT-local.py \
    --run-name "$RUN" --base-checkpoint "$BASE_CKPT" \
    --data-path "$DATA_LOCAL" --work-dir "$WORK_LOCAL" --save-folder "$SAVE_LOCAL" \
    --model-size "$MODEL_SIZE" --seq-len "$SEQ_LEN" --epochs "$EPOCHS" --num-workers 0 --save-checkpoint \
    $MAXARG $WANDB_FLAG
else
  CDM="chunked"; DILARGS=""
  if [ "$VARIANT" = "hier" ]; then
    CDM="hierarchical_dilated"; DILARGS="--dilation-n $DIL_N --dilation-m $DIL_M"
    # DIL_CYCLE set -> rotating "Hierarchical K" (stride resets every L layers); unset -> saturating (legacy).
    [ -n "${DIL_CYCLE:-}" ] && DILARGS="$DILARGS --dilation-cycle $DIL_CYCLE"
  fi
  if [ "$VARIANT" = "random" ]; then
    CDM="random_doc"; DILARGS="--doc-keep-prob ${KEEP:-0.5}"
    # RAND_PER_EX=1 -> a FRESH random graph per example (vs the default: one graph shared by
    # every example). Ablates whether the model needs a stable sparsity structure to learn.
    [ "${RAND_PER_EX:-0}" = "1" ] && DILARGS="$DILARGS --random-doc-per-example"
  fi
  # std -> cross_doc_mode="standard" (plain causal through the DocumentChunkedAttention module): the
  # control that separates "the module/data is broken" from "the mask is too restrictive".
  if [ "$VARIANT" = "std" ]; then CDM="standard"; fi
  # mask-mixing: MIX_MODE=static (fixed fraction of examples trained full-causal) or
  # curriculum (anneal MIX_START_P -> MIX_END_P full-causal fraction). DEFAULT curriculum
  # (mixing rescues restrictive masks + collapses the design hierarchy; pass MIX_MODE=none to opt out).
  MIXMODE="${MIX_MODE:-curriculum}"; MIXARGS="--mix-mode $MIXMODE"
  [ "$MIXMODE" = "static" ]     && MIXARGS="$MIXARGS --standard-mix-prob ${MIX_PROB:-0.10}"
  [ "$MIXMODE" = "curriculum" ] && MIXARGS="$MIXARGS --mix-start-p ${MIX_START_P:-0.8} --mix-end-p ${MIX_END_P:-0.0}"
  # Hybrid full/chunked: FULL_ATTN=layer indices that run PLAIN full (causal) attention while the rest
  # stay chunked (negative counts from the end, -1=last). Empty -> all layers chunked.
  #
  # GOTCHA: `sbatch --export=ALL,FULL_ATTN=9,10,11` SPLITS ON COMMAS -> the job silently receives
  # FULL_ATTN=9 (one layer, not three) and trains the wrong config while looking healthy. Pass a
  # multi-layer list EITHER as a real env var (`FULL_ATTN=9,10,11 sbatch --export=ALL ...`) OR with
  # colons (FULL_ATTN=9:10:11), which we normalize to commas here.
  FULL_ATTN="${FULL_ATTN:-}"; FULL_ATTN="${FULL_ATTN//:/,}"
  FAARG=""; [ -n "$FULL_ATTN" ] && FAARG="--full-attention-layers ${FULL_ATTN}"
  echo "FULL_ATTN='${FULL_ATTN}' (layers running plain causal; empty = all chunked)"
  # wandb now works on the docchunk path too (it previously logged nowhere -- only the `full` arm,
  # which uses the dense script, was ever wired up). WANDB_GROUP groups a whole sweep.
  WBGROUP=""; [ -n "${WANDB_GROUP:-}" ] && WBGROUP="--wandb-group ${WANDB_GROUP}"
  torchrun --nproc_per_node="$NGPU" --master_port="$MASTER_PORT" \
    src/scripts/train/attn_explore/Qwen3-0.6B-docchunk-mask-mix-contradiction-SFT-local.py \
    --run-name "$RUN" --base-checkpoint "$BASE_CKPT" \
    --data-dir "$DATA_LOCAL" --work-dir "$WORK_LOCAL" --save-folder "$SAVE_LOCAL" \
    --model-size "$MODEL_SIZE" --seq-len "$SEQ_LEN" --epochs "$EPOCHS" --grad-accum 8 $MIXARGS \
    --cross-doc-mode "$CDM" $DILARGS $FAARG --save-checkpoint --no-compile \
    --seed "${SEED:-0}" $WANDB_FLAG $WBGROUP \
    $MAXARG
fi
echo "=== DONE $(date '+%F %T')  rc=$? ==="
