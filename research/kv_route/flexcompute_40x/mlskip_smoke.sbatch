#!/bin/bash
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:1
#SBATCH --nodelist=cubbins
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --job-name=mlskip_smoke
#SBATCH --output=/data/prasann/logs/mlskip_smoke_%j.log
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || git -C "${SLURM_SUBMIT_DIR:-.}" rev-parse --show-toplevel)}
REPO=$REPO_ROOT
source /data/prasann/node_local_env.sh
export PYTHONPATH=$REPO/src
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
cd "$REPO"
echo "=== A: router overhead (smoke, 2048 tok, 2 rounds) ==="
$PY -u research/kv_route/flexcompute_40x/mlskip_router_overhead.py --scale 4B --tokens 2048 \
    --rounds 2 --samples 2 --spans 1,2 --out /tmp/mlskip_a_smoke.json
echo "=== A rc=$? ==="
echo "=== B: static prune (smoke, 2048 tok, 2 rounds) ==="
$PY -u research/kv_route/flexcompute_40x/mlskip_static_prune.py --scale q3-4B --tokens 2048 \
    --rounds 2 --samples 2 --out /tmp/mlskip_b_smoke.json
echo "=== B rc=$? ==="
echo "=== DONE ==="
