#!/bin/bash
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=01:30:00
#SBATCH --job-name=runghist
#SBATCH --output=/data/prasann/logs/runghist_%j.log
set -uo pipefail
REPO_ROOT=${REPO_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || git -C "${SLURM_SUBMIT_DIR:-.}" rev-parse --show-toplevel)}
REPO=$REPO_ROOT
source /data/prasann/node_local_env.sh
export PYTHONPATH=$REPO/src
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
cd "$REPO"
R=/data/prasann/ffnmoe_exp/runs
for v in v11-t05-L0-d6-warmv10 v12-t02-L0-d6-warmv10; do
  echo "=== $v ==="
  $PY -u research/kv_route/flexcompute_40x/dump_rung_histograms.py \
      --ckpt "$R/q4b-ffnmoe-$v" \
      --data /data/prasann/ffnmoe_exp/contra_n100_qboth_train \
      --out research/kv_route/flexcompute_40x/rungs_$v.json || echo "FAILED $v"
done
echo "=== DONE ==="
