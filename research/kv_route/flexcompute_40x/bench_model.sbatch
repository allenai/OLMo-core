#!/bin/bash
# Full-model step benchmark with initialised weights, so the dense-vs-gated loss comparison is
# meaningful and not NaN==NaN.
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --job-name=headgate-model
#SBATCH --output=/data/prasann/logs/headgate_model_%j.log

set -uo pipefail
REPO_ROOT=${REPO_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || git -C "${SLURM_SUBMIT_DIR:-.}" rev-parse --show-toplevel)}
REPO=$REPO_ROOT
source /data/prasann/node_local_env.sh
export PYTHONPATH=$REPO/src
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
cd "$REPO"
echo "=== node $(hostname) $(date) ==="

echo "=== 0.8B ==="
$PY -u research/kv_route/flexcompute_40x/bench_head.py --bench model --scale 0.8B \
    --lengths 2048,8192,32768 --out research/kv_route/flexcompute_40x/bench_model_0.8B_unfusedfix.json
echo "=== 4B ==="
$PY -u research/kv_route/flexcompute_40x/bench_head.py --bench model --scale 4B \
    --lengths 2048,8192 --out research/kv_route/flexcompute_40x/bench_model_4B_unfusedfix.json
echo "=== DONE $(date) ==="
