#!/bin/bash
# Native (no-HF-export) eval of an olmo-core distcp checkpoint on RULER + contradiction + NQ
# + held-out retrieval generalization probes (hpqa/fiqa/msmarco/scifact). 1 GPU.
#   sbatch --job-name=gen-base --nodelist=sneetches --qos=preemptive --gres=gpu:H200:1 \
#     --export=ALL,MODEL_PATH=/scratch/.../q4b-dense-cpt-step2385-modelonly,OUT_NAME=q4b-cptbase_gen,TOK=Qwen/Qwen3-4B \
#     src/scripts/train/sft/eval_gen_native.sbatch
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/.genbase_%x_%j.log
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
CR=/scratch/users/prasann/corpus-reasoning
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH}"
OUT_NAME="${OUT_NAME:?set OUT_NAME}"
TOK="${TOK:-Qwen/Qwen3-4B}"
MAXLEN="${MAXLEN:-16384}"

source /system/linux/miniforge-3.13/etc/profile.d/conda.sh
OLMOENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$OLMOENV" ] || OLMOENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
conda activate "$OLMOENV"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
# working-tree olmo_core (has partial_rotary_factor) MUST shadow the editable scratch install
export PYTHONPATH="$REPO/src:$CR"
cd "$CR"
echo "HOST=$(hostname) CUDA=$CUDA_VISIBLE_DEVICES MODEL=$MODEL_PATH START=$(date '+%F %T')"
python -u scripts/eval/eval_lc_native.py \
  --model-path "$MODEL_PATH" \
  --out "outputs/eval_results/${OUT_NAME}_native.json" \
  --tokenizer "$TOK" --max-length "$MAXLEN" --ruler-lengths "${RULER_LENGTHS:-L1024,L2048}"
echo "DONE $(date '+%T') -> outputs/eval_results/${OUT_NAME}_native.json"
