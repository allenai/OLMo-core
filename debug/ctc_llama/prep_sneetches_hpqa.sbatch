#!/bin/bash
# Prepare sneetches for the hotpotqa runs: node-local dirs, the marker-repaired distcp base
# (converted in place from the HF dir on /scratch, deterministic given the fixed seed) and the
# tokenized shard pulled from /scratch. Idempotent.
# No `--gres` line: requesting gpu:0 explicitly makes slurm report ReqNodeNotAvail.
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=sneetches
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --job-name=llama-prep-sneetches
#SBATCH --output=/scratch/users/prasann/ctc_llama_logs/prep_sneetches_%j.log
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
LOGDIR=/scratch/users/prasann/ctc_llama_logs
mkdir -p /data/prasann/joblogs /data/prasann/ctc_llama/bases /data/prasann/ctc_llama/shards "$LOGDIR"
ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$ENV" ] || ENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
export PATH="$ENV/bin:$PATH" PYTHONPATH="$REPO/src"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HOME=/data/prasann
echo "=== HOST=$(hostname) START=$(date '+%F %T') ==="
BASE=/data/prasann/ctc_llama/bases/llama32-3b-base-fixmark
if [ -f "$BASE/model_and_optim/.metadata" ]; then
  echo "base already present"
else
  cd "$REPO"
  python src/scripts/train/memexpress/ctc_suite/convert_llama_base.py \
    --base-dir /scratch/users/prasann/hf_models/Llama-3.2-3B \
    --tokenizer /scratch/users/prasann/hf_models/Llama-3.2-3B-marker-tok \
    --out "$BASE" || { echo "BASE CONVERT FAILED"; exit 3; }
  cp -f "$BASE/marker_audit.json" "$LOGDIR/marker_audit_sneetches.json" 2>/dev/null
fi
[ -f "$BASE/model_and_optim/.metadata" ] || { echo "FATAL: base .metadata missing"; exit 3; }
S=/scratch/users/prasann/ctc_llama/shards/hotpotqa_train_llama
L=/data/prasann/ctc_llama/shards/hotpotqa_train_llama
mkdir -p "$L"; cp -f "$S"/* "$L"/ || exit 4
echo "base $(du -sh "$BASE"|cut -f1); shard $(du -sh "$L"|cut -f1)"; cat "$L/metadata.json"
touch "$LOGDIR/SNEETCHES_READY"
echo "=== PREP DONE $(date '+%F %T') ==="
