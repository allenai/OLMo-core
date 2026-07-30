#!/bin/bash
# Export the DENSE base CPT distcp -> HF, then eval_lc_fast full suite (RULER L1024/L2048 + 5 tasks
# + gen probes) -- the "before" row for the dense before/after table (same eval as the FT runs).
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/.densebase_%j.log
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
CR=/scratch/users/prasann/corpus-reasoning
BASE="${MODEL_PATH:-/scratch/users/prasann/cpt_mix_ckpts/q4b-dense-cpt-step2385-modelonly}"
OUT_NAME="${OUT_NAME:-q4b-dense-base}"
HF_OUT=/data/prasann/cpt_mix_hf/${OUT_NAME}_hf
RESULT=$CR/outputs/eval_results/${OUT_NAME}_lc.json
LADDER_ARGS="${LADDER_ARGS:-}"
source /system/linux/miniforge-3.13/etc/profile.d/conda.sh
OLMOENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$OLMOENV" ] || OLMOENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
EVALENV=/scratch/users/prasann/conda/envs/corpus-reasoning-eval
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
echo "=== [1/2] EXPORT dense base -> HF $(date '+%T') ==="
conda activate "$OLMOENV"; export PYTHONPATH="$CR"; cd "$CR"
python -u scripts/train/export_olmo_to_hf.py --save-folder "$BASE" --ckpt "$BASE" \
  --hf-out "$HF_OUT" --base-model Qwen/Qwen3-4B-Base
erc=$?; echo "export rc=$erc"; [ $erc -ne 0 ] && { echo "EXPORT FAILED"; exit $erc; }
conda deactivate
echo "=== [2/2] EVAL $(date '+%T') ==="
conda activate "$EVALENV"; export PYTHONPATH="$CR"
python -u scripts/eval/eval_lc_fast.py --base-model "$HF_OUT" --out "$RESULT" \
  --max-test-samples 100 --ruler-lengths "${RULER_LENGTHS:-L1024,L2048}" --max-model-len "${MAXLEN:-16384}" $LADDER_ARGS
echo "=== DONE $(date '+%T') result=$RESULT ==="
