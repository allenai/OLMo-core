# Make a CUDA 12 toolchain visible to tilelang's JIT.
#
# Sourced by BOTH build_hils_env_weka.sh and hils_env_setup.sh, so the env is built and used with
# identical paths. In: an ACTIVE venv (+ optionally $CUDA12_PREFIX). Out: CUDA_HOME, PATH,
# LD_LIBRARY_PATH.
#
# tilelang JIT-compiles every kernel at runtime, so it needs BOTH:
#
#  1. libnvrtc.so.12 on the loader path. It IS installed -- torch's cu128 wheels vendor it under
#     site-packages/nvidia -- but nothing adds that directory to LD_LIBRARY_PATH. Missing it
#     surfaces as `ValueError: No CUDA or HIP or MPS available on this system` on a working H100.
#
#  2. A real `nvcc`. This beaker image is a runtime image: no nvcc on PATH, no /usr/local/cuda. And
#     the obvious pip fix does not work -- tilelang's own env.py says so:
#         "from pypi package nvidia-cuda-nvcc, only nvidia-cuda-nvcc>=13.0 works.
#          nvidia-cuda-nvcc-cu12, etc. only installs `ptxas`, not `nvcc`"
#     Verified: the nvidia-cuda-nvcc-cu12 12.9.86 wheel ships exactly one binary, ptxas. Taking the
#     >=13.0 route would put a CUDA 13 compiler in front of a CUDA 12.8 stack (torch 2.8.0+cu128,
#     driver 570), so instead build_hils_env_weka.sh unpacks NVIDIA's CUDA 12.8 redist into
#     $CUDA12_PREFIX and this resolves to it.
#
# Resolution order for CUDA_HOME: an explicit one wins, then a system toolkit, then ours.
_nvdir=$(python -c "import nvidia, os; print(os.path.dirname(nvidia.__file__))" 2>/dev/null)
if [ -n "${_nvdir:-}" ] && [ -d "$_nvdir" ]; then
  for _sub in cuda_nvrtc cuda_runtime cublas cuda_cupti nvjitlink; do
    [ -d "$_nvdir/$_sub/lib" ] && export LD_LIBRARY_PATH="$_nvdir/$_sub/lib:${LD_LIBRARY_PATH:-}"
  done
else
  echo "[cuda-paths] WARNING: no nvidia/ wheel dir in site-packages; libnvrtc may be unreachable."
fi

CUDA12_PREFIX="${CUDA12_PREFIX:-/weka/oe-training-default/amandab/envs/cuda12}"
_cuda_home=""
for _cand in "${CUDA_HOME:-}" "/usr/local/cuda" "$CUDA12_PREFIX"; do
  [ -n "$_cand" ] && [ -x "$_cand/bin/nvcc" ] && { _cuda_home="$_cand"; break; }
done
if [ -z "$_cuda_home" ] && command -v nvcc >/dev/null 2>&1; then
  _cuda_home=$(dirname "$(dirname "$(command -v nvcc)")")
fi
if [ -n "$_cuda_home" ]; then
  export CUDA_HOME="$_cuda_home"
  export PATH="$CUDA_HOME/bin:$PATH"
  [ -d "$CUDA_HOME/lib64" ] && export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
  echo "[cuda-paths] CUDA_HOME=$CUDA_HOME nvcc=$(nvcc --version 2>/dev/null | tail -1 | tr -s ' ')"
else
  echo "[cuda-paths] WARNING: no nvcc found (looked in \$CUDA_HOME, /usr/local/cuda, $CUDA12_PREFIX," \
       "and PATH). tilelang JIT will fail -- run build_hils_env_weka.sh to install one."
fi
unset _nvdir _sub _cand _cuda_home
