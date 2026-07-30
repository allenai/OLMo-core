#!/bin/bash
# Convert the released HF Olmo-3 7B base (allenai/Olmo-3-1025-7B) -> olmo-core distcp base for the
# CTC-suite OLMo arm, then AUDIT the document-marker embedding rows (cosine AND norm -- both known
# marker bugs, see records/document-chunked-marker-embeddings.md + n100-chunked-marker-position-bug.md).
#
# Marker ids for the "olmo3" family are the dolma2 tokenizer's <|extra_id_1|>/<|extra_id_2|> rows
# (100266/100267), exposed under the project's canonical <|box_start|>/<|box_end|> spellings by the
# renamed tokenizer dir at /scratch/users/prasann/hf_models/Olmo-3-1025-7B-docchunk. No vocab growth,
# no embedding resize.
#
#   sbatch --partition=jsteinhardt --qos=preemptive --account=site -w mooney \
#     src/scripts/train/memexpress/ctc_suite/convert_olmo3_base.sbatch
#SBATCH --job-name=olmo3-convert
#SBATCH --nodes=1
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=/data/prasann/joblogs/olmo3_convert_%j.log

set -uo pipefail
MIRROR=/scratch/users/prasann/ctc_suite_logs
mkdir -p "$MIRROR" 2>/dev/null || true
_SRC_LOG="/data/prasann/joblogs/olmo3_convert_${SLURM_JOB_ID}.log"
_DST_LOG="$MIRROR/olmo3_convert_${SLURM_JOB_ID}.log"
( while true; do cp -f "$_SRC_LOG" "$_DST_LOG" 2>/dev/null; sleep 20; done ) &
_MP=$!
trap 'cp -f "$_SRC_LOG" "$_DST_LOG" 2>/dev/null; kill $_MP 2>/dev/null' EXIT

REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$ENV" ] || ENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$REPO/src"
export PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HOME=/data/prasann/home_cache
mkdir -p "$HOME" 2>/dev/null || true

HF_SRC=/scratch/users/prasann/hf_models/Olmo-3-1025-7B
OUT=${OUT:-/scratch/users/prasann/cpt_mix_ckpts/olmo3-7b-base-modelonly}

echo "=== HOST=$(hostname) convert $HF_SRC -> $OUT  START=$(date '+%F %T') ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -u "$REPO/src/examples/huggingface/convert_checkpoint_from_hf.py" \
  -i "$HF_SRC" -o "$OUT" --model-arch olmo3_7b --tokenizer dolma2 \
  --device cuda --validation-device cuda
RC=$?
echo "=== convert rc=$RC $(date '+%F %T') ==="
[ $RC -ne 0 ] && exit $RC

ls -la "$OUT" "$OUT/model_and_optim" | head -20
[ -f "$OUT/model_and_optim/.metadata" ] || { echo "FATAL: no .metadata"; exit 3; }

echo "=== MARKER AUDIT ==="
python -u "$REPO/src/scripts/train/memexpress/ctc_suite/olmo3_marker_audit.py" \
  --checkpoint "$OUT/model_and_optim" --out "$OUT/marker_audit.json"
echo "=== DONE $(date '+%F %T') ==="
