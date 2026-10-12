#!/bin/bash
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:1
#SBATCH --nodelist=cubbins
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=04:00:00
#SBATCH --job-name=steppart_scale
#SBATCH --output=/data/prasann/logs/steppart_scale_%j.log
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || git -C "${SLURM_SUBMIT_DIR:-.}" rev-parse --show-toplevel)}
REPO=$REPO_ROOT
source /data/prasann/node_local_env.sh
export PYTHONPATH=$REPO/src
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=/data/prasann/tmp
mkdir -p "$TMPDIR"
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
cd "$REPO"
B=research/kv_route/flexcompute_40x/bench_step_partition.py

# 27B first: it is the one that may not fit, and a preemption should not cost us the answer.
echo "### 27B ac"   && $PY -u $B --scale 27B  --ac --tokens 8192,16384
echo "### 9B ac"    && $PY -u $B --scale 9B   --ac --tokens 8192,16384
echo "### 9B noac"  && $PY -u $B --scale 9B        --tokens 8192,16384
echo "### 4B ac"    && $PY -u $B --scale 4B   --ac --tokens 8192,16384
echo "### 0.8B ac"  && $PY -u $B --scale 0.8B --ac --tokens 8192,16384
echo "=== DONE ==="
