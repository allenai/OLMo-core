#!/bin/bash
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:1
#SBATCH --nodelist=cubbins
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --job-name=graphnosync
#SBATCH --output=/data/prasann/logs/graphnosync_%j.log
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || git -C "${SLURM_SUBMIT_DIR:-.}" rev-parse --show-toplevel)}
REPO=$REPO_ROOT
source /data/prasann/node_local_env.sh
export PYTHONPATH=$REPO/src
cd "$REPO"
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
S=${SCALE:-4B}; C=${CTX:-16384}
for ARM in staticFFN+bskipK-0.25 staticFFN+bskipK-0.0625 J-f1.00-k0.2500; do
  echo "### nosync $ARM"
  $PY -u research/kv_route/flexcompute_40x/graph_probe.py --case model_fwd_bwd_statichead \
      --scale "$S" --ctx "$C" --head-capacity 256 --routed "$ARM" --bskip-nosync 2>&1 \
    | grep -E "^(RESULT|[A-Za-z]*Error)" | tail -3
done
echo "=== DONE ==="
