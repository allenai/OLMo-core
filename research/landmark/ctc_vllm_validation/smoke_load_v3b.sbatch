#!/bin/bash
#SBATCH --job-name=vllm-smoke-v3b
#SBATCH --account=site
#SBATCH --partition=berkeleynlp
#SBATCH --qos=preemptive
#SBATCH --nodelist=horton
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --output=/data/prasann/joblogs/vllm_smoke_v3b_%j.log
set -uo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=/data/prasann/ctc_vllm_venv/bin/python
REPO_ROOT=${REPO_ROOT:-$(git -C "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" rev-parse --show-toplevel)}
VDIR=$REPO_ROOT/research/landmark/ctc_vllm_validation
SERVE=/data/prasann/ctc_suite/vllm_serving_4b_v3/ctc-s5-contra-full-4b

echo "=== HOST=$(hostname) START=$(date '+%F %T') ==="
echo "=== graft dummy visual weights ==="
$PY -u "$VDIR/add_dummy_visual.py" --serve "$SERVE" || { echo "GRAFT FAILED"; exit 1; }

echo "=== vLLM load + generate smoke ==="
$PY -u "$VDIR/smoke_load_v3.py"
echo "=== DONE $(date '+%F %T') ==="
