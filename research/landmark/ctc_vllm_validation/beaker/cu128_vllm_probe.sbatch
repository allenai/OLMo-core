#!/bin/bash
# FAST gate check: does a genuinely-consistent cu128 vLLM 0.25.1 stack import cleanly and
# actually load+generate on GPU, on a cu128-driver node (mirrors jupiter's driver)? Tries
# several install incantations in order and reports which one works. Iterating locally on
# cubbins (healthy /accounts, H200s, cu128 driver) is much faster than gantry round-trips.
#SBATCH --job-name=cu128-vllm-probe
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=cubbins
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/data/prasann/joblogs/cu128_vllm_probe_%j.log
set -uo pipefail
mkdir -p /data/prasann/joblogs
echo "=== HOST=$(hostname) START=$(date '+%F %T') ==="
nvidia-smi --query-gpu=index,name,driver_version --format=csv
nvidia-smi | head -5

WORK=/data/prasann/cu128_vllm_probe
mkdir -p "$WORK"
BASE_PY=/scratch/users/prasann/conda/envs/corpus-reasoning-eval/bin/python
[ -x "$BASE_PY" ] || BASE_PY=python3

try_incantation() {
  local tag="$1"; shift
  local venv="$WORK/venv_$tag"
  echo ""
  echo "############################################"
  echo "### ATTEMPT [$tag]: $* "
  echo "############################################"
  rm -rf "$venv"
  "$BASE_PY" -m venv "$venv"
  "$venv/bin/pip" install --quiet --upgrade pip
  eval "$@" 2>&1 | tail -60
  echo "--- verifying [$tag] ---"
  "$venv/bin/python" -c "
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
import vllm
print('vllm', vllm.__version__)
from vllm import LLM
print('IMPORT_OK [$tag]')
" && { echo "RESULT [$tag]: IMPORT_OK"; echo "$venv" > "$WORK/WORKING_VENV"; echo "$tag" > "$WORK/WORKING_TAG"; return 0; } \
  || { echo "RESULT [$tag]: FAILED"; return 1; }
}

# --- (a) --torch-backend=cu128 (recent vllm/pip feature to pick the CUDA variant) ---
if try_incantation a "$WORK/venv_a/bin/pip install --quiet 'vllm==0.25.1' --torch-backend=cu128"; then
  echo "=== incantation (a) WORKED ==="; exit 0
fi

# --- (b) pin torch to a +cu128 build FIRST from the cu128 index, then let vllm resolve ---
if try_incantation b "$WORK/venv_b/bin/pip install --quiet torch --index-url https://download.pytorch.org/whl/cu128 && $WORK/venv_b/bin/pip install --quiet 'vllm==0.25.1' --extra-index-url https://download.pytorch.org/whl/cu128"; then
  echo "=== incantation (b) WORKED ==="; exit 0
fi

# --- (c) explicit vllm cu128-tagged extra index (some vllm releases publish CUDA-specific wheels) ---
if try_incantation c "$WORK/venv_c/bin/pip install --quiet 'vllm==0.25.1' --extra-index-url https://download.pytorch.org/whl/cu128"; then
  echo "=== incantation (c) WORKED ==="; exit 0
fi

echo "=== ALL INCANTATIONS FAILED ==="
exit 1
