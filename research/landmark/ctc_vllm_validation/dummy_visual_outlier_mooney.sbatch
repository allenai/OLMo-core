#!/bin/bash
# Piece 3 of the 7-piece recipe: graft dummy visual.* weights (CPU-only, needs the vLLM
# venv's native-qwen3_5 transformers). Separate from the GPU eval job so we don't hold
# the H200 during this step either.
#SBATCH --job-name=dummy-visual-outlier-mooney
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=mooney
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/data/prasann/joblogs/dummy_visual_outlier_mooney_%j.log
set -uo pipefail

SERVE=/data/prasann/ctc_suite/vllm_serving_4b/ctc-4b-outlier-full
VLLM_PY=/data/prasann/ctc_vllm_venv/bin/python
REPO_ROOT=${REPO_ROOT:-$(git -C "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" rev-parse --show-toplevel)}
VDIR=$REPO_ROOT/research/landmark/ctc_vllm_validation

echo "=== HOST=$(hostname) START=$(date '+%F %T') ==="
"$VLLM_PY" -c "import vllm; print('vllm', vllm.__version__)"

NEED_DUMMY_VISUAL=1
if "$VLLM_PY" -c "
from safetensors import safe_open
with safe_open('$SERVE/model.safetensors', framework='pt') as f:
    keys = f.keys()
import sys
sys.exit(0 if any(k.startswith('visual.') for k in keys) else 1)
"; then
  NEED_DUMMY_VISUAL=0
fi

if [ "$NEED_DUMMY_VISUAL" = "1" ]; then
  echo "=== grafting dummy visual weights $(date '+%F %T') ==="
  "$VLLM_PY" "$VDIR/add_dummy_visual.py" --serve "$SERVE"
  rc=$?
  if [ $rc -ne 0 ]; then echo "!!! add_dummy_visual FAILED rc=$rc"; exit 1; fi
else
  echo "visual.* weights already present in $SERVE/model.safetensors"
fi
echo "=== DONE $(date '+%F %T') ==="
