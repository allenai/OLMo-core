#!/bin/bash
# CPU-only diagnostic (no GPU needed): read the EXACT version-guard macros flashinfer 0.6.13's
# bundled cccl checks (cuda_toolkit.h:41 "CUDA compiler and CUDA toolkit headers are
# incompatible") vs what nvcc==13.2.86 (from cuda-toolkit==13.0.2) actually reports, to get
# concrete numbers instead of guessing at another pip pin.
set -uo pipefail
echo "=== HOST=$(hostname) START=$(date '+%F %T') ==="
WORK=/root/nvcc_cccl_diag
mkdir -p "$WORK"
VENV="$WORK/venv"
python3 -m venv "$VENV"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet "flashinfer-python==0.6.13" "flashinfer-cubin==0.6.13" "cuda-toolkit==13.0.2" 2>&1 | tail -60
echo "=== installed versions ==="
"$VENV/bin/pip" list 2>/dev/null | grep -E -i "nvidia-cuda|flashinfer"

NVCC_PATH=$(find "$VENV" -iname nvcc 2>/dev/null | head -1)
echo "NVCC_PATH=$NVCC_PATH"
"$NVCC_PATH" --version

echo "=== nvcc's own __CUDACC_VER_* macros ==="
echo | "$NVCC_PATH" -x cu -E -dM - 2>/dev/null | grep -E "CUDACC_VER|__CUDA_API_VER"

CCCL_HDR=$(find "$VENV" -path '*flashinfer/data/cccl*' -iname 'cuda_toolkit.h' 2>/dev/null | head -1)
echo "CCCL_HDR=$CCCL_HDR"
echo "=== cuda_toolkit.h relevant excerpt (guard logic) ==="
sed -n '1,60p' "$CCCL_HDR"

echo "=== searching this header + its includes for CCCL_VERSION / _LIBCUDACXX_CUDA_API_VERSION / toolkit version macros ==="
grep -rEn "CCCL_VERSION|_LIBCUDACXX_CUDA_API_VERSION|CUDA_TOOLKIT_VERSION|__CUDACC_VER" "$(dirname "$CCCL_HDR")" 2>/dev/null | head -40

# Also check what CUDA toolkit "version.json" or similar the nvcc install itself declares,
# to see if it's self-consistent (nvcc pkg version 13.2.86 vs the actual toolkit release it
# reports internally via __CUDACC_VER_MAJOR__/MINOR__/BUILD__).
NVCC_INCLUDE_DIR=$(dirname "$(dirname "$NVCC_PATH")")/include
echo "NVCC_INCLUDE_DIR=$NVCC_INCLUDE_DIR"
find "$NVCC_INCLUDE_DIR" -maxdepth 1 -iname "cuda.h" -o -iname "*version*" 2>/dev/null
if [ -f "$NVCC_INCLUDE_DIR/cuda.h" ]; then
  grep -E "CUDA_VERSION" "$NVCC_INCLUDE_DIR/cuda.h" | head -5
fi

echo "=== DONE $(date '+%F %T') ==="
