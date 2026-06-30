#!/bin/bash
# LADDER eval (2k..32k) for the 4 cptmix-5task-32k SFT runs (dense / landmark / 2x compressive).
# Fires once a checkpoint is synced LOCAL: weka is NOT mounted on the Berkeley H200s, so first pull
# the model-only step dir weka->s3->/data (see "LADDER EVAL HARNESS" in OVERNIGHT_RESULTS.md).
#
#   MODE=dense    -> scripts/eval/eval_lc_native.py           (the dense run)
#   MODE=landmark -> scripts/eval/eval_lc_native_landmark.py  (plain-landmark + BOTH compressive runs;
#                    native landmark-token insertion at prefill; compressive uses the same eval path)
#
# Both eval scripts support torchrun data-parallel (WORLD_SIZE) -> shard examples across GPUs.
# Speedups already in the scripts: per-row stop-string early-stop + KV-cache decode. Contradiction is
# NO-COT (direct pairs): short --contra-max-new-tokens budget + early-stop on "contradicting pairs:"
# + think-strip + no newline-stop (the ladder loop applies these automatically for contradiction).
#
# Example:
#   sbatch --job-name=lad-dense --nodelist=horton --qos=preemptive_high --gres=gpu:H200:4 \
#     --export=ALL,MODE=dense,MODEL_PATH=/data/prasann/cpt_mix_runs/q4b-cptmix-5task-32k/step1465,OUT_NAME=q4b-dense-32k \
#     src/scripts/train/sft/eval_ladder_32k.sbatch
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/.ladder_%x_%j.log
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
CR=/scratch/users/prasann/corpus-reasoning
MODE="${MODE:?set MODE=dense|landmark}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH (local synced step1465 dir)}"
OUT_NAME="${OUT_NAME:?set OUT_NAME}"
NSAMP="${NSAMP:-500}"               # contra/rerank/nq/outlier rungs all have >=400; oolong capped by file
MAXLEN="${MAXLEN:-49152}"           # fit the 32k rung (~40k content + landmark tokens + generation)
CONTRA_MAXNEW="${CONTRA_MAXNEW:-96}"  # NO-COT direct pairs (short budget)
NPROC="${NPROC:-4}"
TOPK="${TOPK:-0.1}"                  # landmark inference top-k fraction (matches trained sparsity)
MAXBATCH="${MAXBATCH:-49152}"       # landmark length-bucketed batch budget (B*L), no padding
DENSE_BATCH="${DENSE_BATCH:-4}"     # dense HF-export generate_batch: pad-to-longest, so small batch at 32k rung (16 OOMs)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Optional job-splitting / filtering (SLOWNESS mitigation): restrict the ladder to a subset of
# tasks/rungs so each job fits the wall-clock cap, and skip RULER/gen probes when merging later.
LADDER_TASKS="${LADDER_TASKS:-}"   # e.g. oolong  OR  rerank,outlier
LADDER_RUNGS="${LADDER_RUNGS:-}"   # e.g. 16k,32k
SKIP_RULER="${SKIP_RULER:-0}"
SKIP_GEN="${SKIP_GEN:-0}"
EXTRA=()
[ -n "$LADDER_TASKS" ] && EXTRA+=(--ladder-tasks "$LADDER_TASKS")
[ -n "$LADDER_RUNGS" ] && EXTRA+=(--ladder-rungs "$LADDER_RUNGS")
[ "$SKIP_RULER" = 1 ] && EXTRA+=(--skip-ruler)
[ "$SKIP_GEN" = 1 ] && EXTRA+=(--skip-gen)
source /system/linux/miniforge-3.13/etc/profile.d/conda.sh
OLMOENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$OLMOENV" ] || OLMOENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
conda activate "$OLMOENV"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/src:$CR"
cd "$CR"
MASTER_PORT="${MASTER_PORT:-$((20000 + (RANDOM % 20000)))}"   # avoid 29500 collision when co-located
echo "HOST=$(hostname) MODE=$MODE MODEL=$MODEL_PATH NSAMP=$NSAMP MAXLEN=$MAXLEN CONTRA_MAXNEW=$CONTRA_MAXNEW NPROC=$NPROC MASTER_PORT=$MASTER_PORT START=$(date '+%T')"
if [ "$MODE" = dense ]; then
  torchrun --nproc_per_node=$NPROC --master_port=$MASTER_PORT scripts/eval/eval_lc_native.py \
    --model-path "$MODEL_PATH" --out "outputs/eval_results/${OUT_NAME}_ladder.json" \
    --tokenizer Qwen/Qwen3-4B --ladder --max-test-samples "$NSAMP" \
    --max-length "$MAXLEN" --contra-max-new-tokens "$CONTRA_MAXNEW" \
    --batch-size "$DENSE_BATCH" \
    --ruler-lengths L1024,L2048 "${EXTRA[@]}"
else
  torchrun --nproc_per_node=$NPROC --master_port=$MASTER_PORT scripts/eval/eval_lc_native_landmark.py \
    --model-path "$MODEL_PATH" --out "outputs/eval_results/${OUT_NAME}_ladder.json" \
    --tokenizer Qwen/Qwen3-4B --ladder --max-test-samples "$NSAMP" \
    --max-length "$MAXLEN" --contra-max-new-tokens "$CONTRA_MAXNEW" \
    --landmark-top-k-fraction "$TOPK" --max-batch-tokens "$MAXBATCH" \
    --ruler-lengths L1024,L2048 "${EXTRA[@]}"
fi
echo "DONE $(date '+%T') -> outputs/eval_results/${OUT_NAME}_ladder.json"
