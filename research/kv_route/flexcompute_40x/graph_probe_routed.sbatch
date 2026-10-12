#!/bin/bash
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:1
#SBATCH --nodelist=cubbins
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --job-name=graphrouted
#SBATCH --output=/data/prasann/logs/graphrouted_%j.log
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || git -C "${SLURM_SUBMIT_DIR:-.}" rev-parse --show-toplevel)}
REPO=$REPO_ROOT
source /data/prasann/node_local_env.sh
export PYTHONPATH=$REPO/src
cd "$REPO"
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
run () {  # one case per PROCESS: a failed capture poisons every later capture in the context
  echo "### $*"
  $PY -u research/kv_route/flexcompute_40x/graph_probe.py "$@" 2>&1 \
    | grep -E "^(RESULT|#|[A-Za-z]*Error)" | tail -4
}
S=${SCALE:-4B}; C=${CTX:-16384}
# unrouted ladder at the real scale: confirm the fused-CE blocker is not a 0.8B artefact
for CASE in model_fwd_bwd_nohead model_fwd_bwd_defaultce model_fwd_bwd_fusedce \
            model_fwd_bwd_gatedhead model_fwd_bwd_statichead; do
  run --case "$CASE" --scale "$S" --ctx "$C" --head-capacity 256
done
# the actual question: a graph on top of the routing win
for ARM in dense staticFFN+bskipK-0.25 J-f1.00-k0.2500 staticFFN+bskipK-0.0625; do
  run --case model_fwd_bwd_statichead --scale "$S" --ctx "$C" --head-capacity 256 --routed "$ARM"
done
echo "=== DONE ==="
