#!/bin/bash
# Repair the reserved-token embedding rows of the Qwen3.5-4B converted base (esp. the landmark row
# 248200, needed by the sparselandmark arm) -> q35-4b-base-markerfix on cubbins /data.
# The "before" norms in the log document whether the base was already repaired.
#SBATCH --job-name=q35-markerfix
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=02:00:00
#SBATCH --nodelist=cubbins
#SBATCH --output=/data/prasann/joblogs/q35_markerfix_%j.log
#SBATCH --chdir=/data/prasann
set -uo pipefail
export HOME=/data/prasann/home
export HF_HOME=/data/prasann/hf_cache
export TMPDIR=/data/prasann/tmp
mkdir -p /data/prasann/joblogs "$TMPDIR"

REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
[ -x "$PY" ] || PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python

echo "=== [q35-markerfix] HOST=$(hostname) START=$(date '+%F %T') python=$PY ==="
PYTHONPATH="$REPO/src" $PY "$REPO/src/scripts/data/fix_marker_embeddings_qwen35.py" \
  --base /data/prasann/ctc_suite/bases/q35-4b-base-modelonly/model_and_optim \
  --out  /data/prasann/ctc_suite/bases/q35-4b-base-markerfix \
  --model-scale 4b || { echo "!!! repair FAILED"; exit 1; }
echo "=== [q35-markerfix] DONE $(date '+%F %T') ==="
