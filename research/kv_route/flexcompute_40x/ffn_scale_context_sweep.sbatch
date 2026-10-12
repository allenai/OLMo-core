#!/bin/bash
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:1
#SBATCH --nodelist=cubbins
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --job-name=ffnctx
#SBATCH --output=/data/prasann/logs/ffnctx_%j.log
# The FFN router's FLOP->time conversion is strongly context dependent (4B: 4.3x at 8k vs 19.2x at
# 32k for the same mean cost). The ceiling model needs that curve at every scale, not just 4B, or
# the context axis of the ceiling table is an extrapolation off one model size.
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
for S in 0.8B 4B 9B 27B; do
  for T in 2048 4096 16384 65536 131072; do
    echo "### $S $T"
    $PY -u research/kv_route/flexcompute_40x/ffn_null_rung_sweep.py --scale "$S" --tokens "$T"
  done
done
echo "=== DONE ==="
