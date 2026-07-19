#!/bin/bash
# One-off: convert the Qwen3.5-{4B,9B}-Base HF checkpoints (already staged node-local on cubbins
# at /data/prasann/hf_models/Qwen3.5-{4B,9B}-Base) to olmo-core model-only distcp bases for the
# CTC-suite scaling plan (records/ctc-suite-scaling-plan.md Stage 0). CPU-only (the hybrid
# Qwen3.5 convert path builds on init_device="cpu" and never touches cuda) -- no GPU requested.
#
#   sbatch --export=ALL,SCALE=4b convert_qwen35_bases.sbatch
#   sbatch --export=ALL,SCALE=9b convert_qwen35_bases.sbatch
#SBATCH --job-name=q35-convert
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=cubbins
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/data/prasann/joblogs/q35_convert_%x_%j.log

set -uo pipefail
REPO="${REPO:-/accounts/projects/berkeleynlp/prasann/projects/OLMo-core}"
SCALE="${SCALE:?set SCALE=4b|9b}"
BASE_DIR="${BASE_DIR:-/data/prasann/hf_models/Qwen3.5-${SCALE^^}-Base}"
OUT="${OUT:-/data/prasann/ctc_suite/bases/q35-${SCALE}-base-modelonly}"
echo "=== HOST=$(hostname) SCALE=$SCALE BASE_DIR=$BASE_DIR OUT=$OUT START=$(date '+%F %T') ==="

ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$ENV" ] || ENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$REPO/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONWARNINGS=ignore
export PYTHONUNBUFFERED=1

timeout 6000 python "$REPO/src/scripts/train/memexpress/ctc_suite/convert_qwen35_base.py" \
    --base-dir "$BASE_DIR" --out "$OUT"
rc=$?
echo "=== DONE rc=$rc END=$(date '+%F %T') ==="
exit $rc
