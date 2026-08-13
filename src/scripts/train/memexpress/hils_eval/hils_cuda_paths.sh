# Put the CUDA 12 compiler + runtime libraries shipped in pip wheels onto the loader path.
#
# Sourced by BOTH build_hils_env_weka.sh and hils_env_setup.sh, so the env is built and used with
# identical paths. In:  an ACTIVE venv. Out: PATH / LD_LIBRARY_PATH / CUDA_HOME.
#
# tilelang JIT-compiles every kernel at runtime, so it needs nvrtc and nvcc present. In this
# container neither is reachable by default:
#   * `nvcc` is not on PATH at all (the beaker image is a runtime image, not a CUDA devel one)
#   * libnvrtc.so.12 IS installed -- torch's cu128 wheels vendor it under site-packages/nvidia --
#     but nothing adds that directory to the dynamic loader path
# Observed as `OSError: libnvrtc.so.12: cannot open shared object file` from tilelang's
# determine_target(), which surfaces as the far more confusing
# `ValueError: No CUDA or HIP or MPS available on this system` on a node with a working H100
# (job 01KZY6Q3WN2C8MGT3N378MPXHC).
#
# Note the version: tilelang's own `[nvcc]` extra pulls nvidia-cuda-nvcc>=13, i.e. CUDA 13, while
# this stack is CUDA 12.8 throughout (torch 2.8.0+cu128, driver 570). Install the -cu12 wheels.
_nvdir=$(python -c "import nvidia, os; print(os.path.dirname(nvidia.__file__))" 2>/dev/null)
if [ -n "${_nvdir:-}" ] && [ -d "$_nvdir" ]; then
  for _sub in cuda_nvrtc cuda_runtime cublas cuda_cupti nvjitlink; do
    [ -d "$_nvdir/$_sub/lib" ] && export LD_LIBRARY_PATH="$_nvdir/$_sub/lib:${LD_LIBRARY_PATH:-}"
  done
  if [ -d "$_nvdir/cuda_nvcc/bin" ]; then
    export PATH="$_nvdir/cuda_nvcc/bin:$PATH"
    export CUDA_HOME="$_nvdir/cuda_nvcc"
  fi
  echo "[cuda-paths] nvidia wheels at $_nvdir; nvcc=$(command -v nvcc || echo MISSING)"
else
  echo "[cuda-paths] WARNING: no nvidia/ wheel dir found in site-packages; tilelang JIT will fail."
fi
unset _nvdir _sub
