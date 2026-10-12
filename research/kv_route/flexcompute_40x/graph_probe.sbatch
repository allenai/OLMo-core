#!/bin/bash
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:1
#SBATCH --nodelist=cubbins
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --job-name=graphprobe
#SBATCH --output=/data/prasann/logs/graphprobe_%j.log
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || git -C "${SLURM_SUBMIT_DIR:-.}" rev-parse --show-toplevel)}
REPO=$REPO_ROOT
source /data/prasann/node_local_env.sh
export PYTHONPATH=$REPO/src
cd "$REPO"
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
# One case per PROCESS: a failed capture poisons every later capture in the same context, so a
# single-process loop would blame the first failure for all of them.
for CASE in matmul block_fwd block_fwd_bwd model_fwd_nohead model_fwd_bwd_nohead \
            model_fwd_bwd_nohead_nobudget model_fwd_bwd_defaultce model_fwd_bwd_fusedce; do
  echo "### case $CASE"
  $PY -u research/kv_route/flexcompute_40x/graph_probe.py --case "$CASE" \
      --scale "${SCALE:-0.8B}" --ctx "${CTX:-2048}" 2>&1 | grep -E "^(RESULT|#|[A-Za-z]*Error)" | tail -6
done
echo "=== DONE ==="
