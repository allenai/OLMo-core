#!/bin/bash
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:1
#SBATCH --nodelist=cubbins
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --time=04:00:00
#SBATCH --job-name=mlskip
#SBATCH --output=/data/prasann/logs/mlskip_%j.log
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || git -C "${SLURM_SUBMIT_DIR:-.}" rev-parse --show-toplevel)}
REPO=$REPO_ROOT
source /data/prasann/node_local_env.sh
export PYTHONPATH=$REPO/src
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
cd "$REPO"

echo "######## 1. per-block routing overhead + span-K amortisation, 4B @ 8192"
$PY -u research/kv_route/flexcompute_40x/mlskip_router_overhead.py --scale 4B --tokens 8192 \
    --rounds 6 --samples 3 --spans 1,2,4,8
echo "rc=$?"

echo "######## 2. static pruning of always-null layers, Qwen3-4B @ 8192"
$PY -u research/kv_route/flexcompute_40x/mlskip_static_prune.py --scale q3-4B --tokens 8192 \
    --rounds 5 --samples 3
echo "rc=$?"

echo "######## 3. span-K on a real 4B step @ 8192"
$PY -u research/kv_route/flexcompute_40x/mlskip_span_step.py --scale 4B --tokens 8192 \
    --keep 1.0,0.5 --spans 1,2,4,8 --rounds 4 --samples 3
echo "rc=$?"

echo "######## 4. per-block routing overhead at 32k (does the overhead share shrink?)"
$PY -u research/kv_route/flexcompute_40x/mlskip_router_overhead.py --scale 4B --tokens 32768 \
    --rounds 4 --samples 3 --spans 1,2,4
echo "rc=$?"

echo "=== DONE ==="
