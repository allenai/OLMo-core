#!/bin/bash
# One-off: download Qwen/Qwen3.5-2B-Base from HF onto cubbins node-local /data and convert it to an
# olmo-core model-only distcp base for the model-scale study (records/ctc-model-scale-plan.md).
# Mirrors convert_qwen35_bases.sbatch but adds the download step -- unlike the 4B/9B bases, the 2B
# was never staged. Compute nodes DO have internet (verified 2026-08-12), so this is one job.
# CPU-only: the hybrid Qwen3.5 convert path builds on init_device="cpu".
#
#   sbatch debug/ctc_modelscale/fetch_convert_2b.sbatch
#SBATCH --job-name=q35-2b-convert
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=cubbins
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/data/prasann/joblogs/q35_2b_convert_%j.log

set -uo pipefail
REPO="${REPO:-/accounts/projects/berkeleynlp/prasann/projects/OLMo-core}"
BASE_DIR="${BASE_DIR:-/data/prasann/hf_models/Qwen3.5-2B-Base}"
OUT="${OUT:-/data/prasann/ctc_suite/bases/q35-2b-base-modelonly}"
echo "=== HOST=$(hostname) BASE_DIR=$BASE_DIR OUT=$OUT START=$(date '+%F %T') ==="

ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$ENV" ] || ENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$REPO/src"
export PYTHONWARNINGS=ignore
export PYTHONUNBUFFERED=1
# Download stage needs the hub ONLINE; the convert stage re-reads from the local dir.
export HF_HOME=/data/prasann/hf_cache
mkdir -p "$HF_HOME" /data/prasann/joblogs "$(dirname "$BASE_DIR")" "$(dirname "$OUT")"

if [ ! -f "$BASE_DIR/config.json" ]; then
    echo "--- downloading Qwen/Qwen3.5-2B-Base -> $BASE_DIR ---"
    python - <<PY
from huggingface_hub import snapshot_download
p = snapshot_download(
    "Qwen/Qwen3.5-2B-Base",
    local_dir="$BASE_DIR",
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.py", "tokenizer*", "*.jinja"],
    max_workers=8,
)
print("downloaded to", p, flush=True)
PY
    rc=$?
    [ $rc -ne 0 ] && { echo "=== DOWNLOAD FAILED rc=$rc ==="; exit $rc; }
else
    echo "--- $BASE_DIR already present, skipping download ---"
fi
ls -la "$BASE_DIR" | head -20

echo "--- converting ---"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
timeout 6000 python "$REPO/src/scripts/train/memexpress/ctc_suite/convert_qwen35_base.py" \
    --base-dir "$BASE_DIR" --out "$OUT"
rc=$?
echo "=== DONE rc=$rc END=$(date '+%F %T') ==="
exit $rc
