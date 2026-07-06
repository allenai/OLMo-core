#!/bin/bash
# Native landmark eval (length-bucketed batching) on a fast-landmark olmo-core distcp checkpoint.
# Full task suite (RULER L1024/L2048 + contra + nq + oolong/rerank/outlier + gen probes).
#   sbatch --job-name=lmbase --nodelist=cubbins --qos=preemptive_high --gres=gpu:H200:1 \
#     --export=ALL,MODEL_PATH=/scratch/.../q4b-fast-landmark-step2385,OUT_NAME=q4b-lm-base \
#     src/scripts/train/sft/eval_landmark_native.sbatch
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --output=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/.lmbench_%x_%j.log
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
CR=/scratch/users/prasann/corpus-reasoning
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH}"
OUT_NAME="${OUT_NAME:?set OUT_NAME}"
NSAMP="${NSAMP:-100}"
MAXBATCH="${MAXBATCH:-32768}"
source /system/linux/miniforge-3.13/etc/profile.d/conda.sh
OLMOENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$OLMOENV" ] || OLMOENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
conda activate "$OLMOENV"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/src:$CR"
cd "$CR"
echo "HOST=$(hostname) CUDA=$CUDA_VISIBLE_DEVICES MODEL=$MODEL_PATH NSAMP=$NSAMP MAXBATCH=$MAXBATCH START=$(date '+%T')"
python -u scripts/eval/eval_lc_native_landmark.py \
  --model-path "$MODEL_PATH" \
  --out "outputs/eval_results/${OUT_NAME}_lmnative.json" \
  --tokenizer Qwen/Qwen3-4B --max-length 16384 \
  --ruler-lengths L1024,L2048 --max-test-samples "$NSAMP" --max-batch-tokens "$MAXBATCH"
echo "DONE $(date '+%T') -> outputs/eval_results/${OUT_NAME}_lmnative.json"
