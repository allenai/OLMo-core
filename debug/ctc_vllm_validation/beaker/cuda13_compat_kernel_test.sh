#!/bin/bash
# Minimal decisive test: does CUDA-13 forward-compat let a real GPU kernel execute on
# jupiter's R570/12.8 driver? No vllm, no transformers -- just torch + a matmul.
set -uo pipefail
echo "=== HOST=$(hostname) START=$(date '+%F %T') ==="
nvidia-smi --query-gpu=index,name,driver_version --format=csv

WORK=/root/cuda13_kernel_test
mkdir -p "$WORK"
VENV="$WORK/venv"

echo "=== installing CUDA-13 forward-compat libs $(date '+%F %T') ==="
. /etc/os-release; UBU_TAG="ubuntu${VERSION_ID//./}"
apt-get update -qq
apt-get install -y -qq wget gnupg ca-certificates >/dev/null
wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${UBU_TAG}/x86_64/cuda-keyring_1.1-1_all.deb" -O /tmp/cuda-keyring.deb \
  && dpkg -i /tmp/cuda-keyring.deb >/dev/null \
  && apt-get update -qq \
  && apt-get install -y -qq cuda-compat-13-0
COMPAT_DIR=$(dpkg -L cuda-compat-13-0 2>/dev/null | grep 'libcuda\.so' | head -1 | xargs -r dirname)
echo "COMPAT_DIR=$COMPAT_DIR"
export LD_LIBRARY_PATH="$COMPAT_DIR:${LD_LIBRARY_PATH:-}"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

echo "=== building minimal torch-only venv $(date '+%F %T') ==="
python3 -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet "torch==2.11.0" 2>&1 | tail -30
"$VENV/bin/python" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

echo "=== THE TEST: real 4096x4096 matmul on GPU $(date '+%F %T') ==="
"$VENV/bin/python" -c "
import torch
print('cuda.is_available()', torch.cuda.is_available())
x = torch.randn(4096, 4096, device='cuda')
y = (x @ x).sum()
print('KERNEL OK', float(y))
"
rc=$?
echo "=== DONE rc=$rc $(date '+%F %T') ==="
exit $rc
