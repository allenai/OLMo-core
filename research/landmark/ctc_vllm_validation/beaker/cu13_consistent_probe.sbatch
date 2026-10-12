#!/bin/bash
# Verify the EXACT consistent cu13 stack (coordinator-specified) loads+generates on a cu13-driver
# local node (cubbins, driver 580.126.09/CUDA13.0) -- fast local iteration before replicating on
# Beaker/jupiter (which needs CUDA-13 forward-compat on top of this same stack).
#SBATCH --job-name=cu13-consistent-probe
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=cubbins
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/data/prasann/joblogs/cu13_consistent_probe_%j.log
set -uo pipefail
mkdir -p /data/prasann/joblogs
echo "=== HOST=$(hostname) START=$(date '+%F %T') ==="
nvidia-smi --query-gpu=index,name,driver_version --format=csv
nvidia-smi | head -5

WORK=/data/prasann/cu13_consistent_probe
mkdir -p "$WORK"
VENV="$WORK/venv"
BASE_PY=/scratch/users/prasann/conda/envs/corpus-reasoning-eval/bin/python
[ -x "$BASE_PY" ] || BASE_PY=python3

rm -rf "$VENV"
"$BASE_PY" -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip
echo "=== installing consistent cu13 stack (default index) $(date '+%F %T') ==="
"$VENV/bin/pip" install --quiet \
  "torch==2.11.0" "torchvision==0.26.0" "torchaudio==2.11.0" \
  "vllm==0.25.1" "transformers==5.14.1" "flashinfer-python==0.6.13" 2>&1 | tail -80
echo "=== install done $(date '+%F %T') ==="

echo "=== verifying clean imports $(date '+%F %T') ==="
"$VENV/bin/python" -c "
import torch, torchvision, torchaudio
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('torchvision', torchvision.__version__)
print('torchaudio', torchaudio.__version__)
print('cuda.is_available()', torch.cuda.is_available())
import vllm
print('vllm', vllm.__version__)
from transformers import Qwen3_5ForCausalLM
print('Qwen3_5ForCausalLM import OK')
import flashinfer
print('flashinfer', flashinfer.__version__)
x = torch.randn(1024, 1024, device='cuda')
y = (x @ x).sum().item()
print('REAL GPU MATMUL RAN, sum=', y)
"
rc=$?
echo "=== import/matmul verification rc=$rc $(date '+%F %T') ==="

echo "=== nvcc discovery $(date '+%F %T') ==="
NVCC_PATH=$(find "$VENV" -iname nvcc 2>/dev/null)
echo "nvcc candidates: $NVCC_PATH"
for p in $NVCC_PATH; do
  echo "-- $p --"
  "$p" --version 2>&1
done
# also check if nvcc is already usable with no extra install (transitively via torch/vllm cu13 deps)
if [ -z "$NVCC_PATH" ]; then
  echo "no nvcc present yet from the base stack -- installing nvidia-cuda-nvcc-cu13 explicitly"
  "$VENV/bin/pip" install --quiet nvidia-cuda-nvcc-cu13 2>&1 | tail -20
  NVCC_PATH=$(find "$VENV" -iname nvcc 2>/dev/null)
  echo "nvcc after explicit install: $NVCC_PATH"
fi

echo "=== DONE $(date '+%F %T') ==="
