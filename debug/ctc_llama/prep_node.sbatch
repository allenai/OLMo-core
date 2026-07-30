#!/bin/bash
# Prepare a Berkeley node for the Llama CTC runs: node-local dirs, the marker-repaired distcp base
# (converted in place from the HF dir on /scratch), and the tokenized shards staged from /scratch.
# Idempotent -- skips the base conversion if a valid .metadata already exists.
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --job-name=llama-prep
#SBATCH --output=/data/prasann/joblogs/llama_prep_%j.log
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
LOGDIR=/scratch/users/prasann/ctc_llama_logs
H=$(hostname)
mkdir -p /data/prasann/joblogs /data/prasann/ctc_llama/bases /data/prasann/ctc_llama/shards "$LOGDIR"
SRC=/data/prasann/joblogs/llama_prep_${SLURM_JOB_ID}.log
( while true; do cp -f "$SRC" "$LOGDIR/prep_${H}.log" 2>/dev/null; sleep 15; done ) &
trap 'cp -f "$SRC" "$LOGDIR/prep_${H}.log" 2>/dev/null; kill %1 2>/dev/null' EXIT

ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$ENV" ] || ENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
export PATH="$ENV/bin:$PATH" PYTHONPATH="$REPO/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HOME=/data/prasann
echo "=== HOST=$H START=$(date '+%F %T') env=$ENV ==="

BASE=/data/prasann/ctc_llama/bases/llama32-3b-base-fixmark
if [ -f "$BASE/model_and_optim/.metadata" ]; then
  echo "base already present: $BASE"
else
  cd "$REPO"
  python src/scripts/train/memexpress/ctc_suite/convert_llama_base.py \
    --base-dir /scratch/users/prasann/hf_models/Llama-3.2-3B \
    --tokenizer /scratch/users/prasann/hf_models/Llama-3.2-3B-marker-tok \
    --out "$BASE" || { echo "BASE CONVERT FAILED"; exit 3; }
  cp -f "$BASE/marker_audit.json" "$LOGDIR/marker_audit_${H}.json" 2>/dev/null
fi
[ -f "$BASE/model_and_optim/.metadata" ] || { echo "FATAL: base .metadata missing"; exit 3; }
echo "base OK: $(du -sh "$BASE" | cut -f1)"

# shards: wait for the staging job to publish them on /scratch, then pull node-local
STAGE=/scratch/users/prasann/ctc_llama/shards
for i in $(seq 1 120); do
  [ -f "$LOGDIR/STAGE_OUT_DONE" ] && break
  echo "waiting for shard staging ($i)"; sleep 30
done
for d in "$STAGE"/*/; do
  n=$(basename "$d"); L=/data/prasann/ctc_llama/shards/$n
  mkdir -p "$L"; cp -n "$d"/* "$L"/ 2>/dev/null
  echo "shard $n -> $L : $(du -sh "$L" | cut -f1)"; cat "$L/metadata.json" 2>/dev/null | head -30
done
echo "=== PREP DONE $(date '+%F %T') ==="
touch "$LOGDIR/PREP_DONE_${H}"
