#!/bin/bash
# Build the HiLS runtime as a Python 3.11 venv ON WEKA, once, for every eval job to reuse.
#
# Why a separate env at all: the OLMo-core beaker image ships Python **3.12**, and the HiLS stack
# does not run on it --
#   * veomni declares requires-python '<3.12,>=3.11' (pip refuses it even with --no-deps)
#   * tilelang 0.1.9 dies on import under 3.12:
#       AttributeError: attribute '__dict__' of 'type' objects is not writable
# The HiLS project pins python 3.11 + torch 2.8.0, so this reproduces that rather than fighting it.
#
# Why on weka rather than per job: ~2 GB of wheels and a python download. Paying that in each of
# ~18 eval jobs is both slow and a fresh chance for a transient PyPI failure to kill an allocated
# GPU node. The env is built once and mounted at a FIXED path by every job, so the venv's absolute
# shebangs resolve identically everywhere.
#
# RUN THIS ON A GPU NODE (1 GPU is enough). The tilelang version is chosen by actually JIT-compiling
# the HiLS chunk-pool kernel, which needs a device -- see the bisect below.
set -euo pipefail

ENV_ROOT="${ENV_ROOT:-/weka/oe-training-default/amandab/envs}"
HILS_ENV="${HILS_ENV:-$ENV_ROOT/hils-py311}"
# uv downloads a managed CPython. It MUST land on weka too -- the default (~/.local/share/uv) is
# container-local, so the venv would point at an interpreter that does not exist in the next job.
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$ENV_ROOT/pythons}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ENV_ROOT/uv-cache}"

TORCH_VERSION="${TORCH_VERSION:-2.8.0}"          # the HiLS pin
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
TRANSFORMERS_VERSION="${TRANSFORMERS_VERSION:-4.57.3}"   # the HiLS pin
VEOMNI_REF="${VEOMNI_REF:-441e1b2483921e9cfe56c8d97541a23cb4b290a8}"
FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.8.3}"   # the HiLS pin
# Newest first; the first one that COMPILES A REAL KERNEL wins. The HiLS requirements.txt pins a
# git sha instead, which is a 20-40 min source build needing LLVM -- not worth it if a wheel works.
# NOTE the ordering trap: sorted() on PyPI's version strings puts "0.1.13" BEFORE "0.1.6"
# lexicographically, so an earlier version of this list silently omitted the four NEWEST releases
# and bisected only the old ones. These are in real version order.
TILELANG_CANDIDATES="${TILELANG_CANDIDATES:-0.1.13 0.1.12 0.1.11 0.1.10 0.1.9 0.1.8 0.1.6.post2}"
# tilelang declares apache-tvm-ffi>=0.1.11,<0.1.13 -- a range PyPI DOES NOT CONTAIN (it jumps
# 0.1.9 -> 0.1.13.post3). Whatever the resolver picks is therefore untested by tilelang's author,
# and 0.1.13.post3 collides with the tvm bundled in the wheel:
#   ValueError: Type 'ffi.Tensor' already has a registered class
# 0.1.9 was verified to import cleanly (diagnostic 01KZY6Q3WN2C8MGT3N378MPXHC, strategy B), so pin
# it explicitly AFTER tilelang installs its own choice.
TVM_FFI_VERSION="${TVM_FFI_VERSION:-0.1.9}"
# Match the stack: torch is cu128 and the driver is 570 (CUDA 12.8).
CUDA_REDIST_MANIFEST="${CUDA_REDIST_MANIFEST:-12.8.1}"
HILS_REPO="${HILS_REPO:-/tmp/HiLS-Attention}"
HILS_GIT="${HILS_GIT:-https://github.com/abertsch72/HiLS-Attention.git}"

mkdir -p "$ENV_ROOT" "$UV_PYTHON_INSTALL_DIR" "$UV_CACHE_DIR"

command -v uv >/dev/null 2>&1 || {
  echo "[build-env] installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
}
echo "[build-env] uv=$(uv --version)"

# TILELANG_ONLY=1 re-runs ONLY the tilelang bisect against an existing env. Without it, a bisect
# rerun deletes and rebuilds the venv -- which would pull the interpreter out from under any eval
# job currently running against it.
if [ "${TILELANG_ONLY:-0}" = "1" ]; then
  [ -f "$HILS_ENV/bin/activate" ] || { echo "[build-env] FATAL: TILELANG_ONLY=1 but no env at $HILS_ENV"; exit 1; }
  echo "[build-env] TILELANG_ONLY=1 -- reusing $HILS_ENV, bisecting tilelang only"
else
  if [ "${REBUILD:-0}" = "1" ]; then
    echo "[build-env] REBUILD=1 -- removing $HILS_ENV"
    rm -rf "$HILS_ENV"
  fi
  echo "[build-env] creating $HILS_ENV (python 3.11)"
  uv venv --python 3.11 "$HILS_ENV"
fi
# shellcheck disable=SC1091
source "$HILS_ENV/bin/activate"
echo "[build-env] python=$(python -V) at $(which python)"

if [ "${TILELANG_ONLY:-0}" != "1" ]; then
echo "[build-env] torch $TORCH_VERSION from $TORCH_INDEX"
uv pip install "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX"

echo "[build-env] transformers + our eval deps"
# flash-attn IS needed after all, and not for speed. Without it the dense layers fall back to sdpa,
# and transformers then materializes a (B, 1, T, T) attention mask because our prompts are padded:
# at the 32k rung with batch 8 that is ~17 GB per attention call, and the control OOM'd on 80 GB
# H100s within two minutes (job 01KZY8Q1C6CW8N5SGRQCAMMA20). FA2 takes the varlen path and never
# builds the mask. Use the PREBUILT wheel for this exact torch/python/ABI -- building from source
# takes 30+ minutes per attempt.
# nvidia-cuda-nvrtc-cu12 only (torch vendors it already; belt and braces). The nvcc COMPILER does
# not come from pip -- see the CUDA 12 redist block below for why.
uv pip install \
  "transformers==$TRANSFORMERS_VERSION" \
  einops accelerate safetensors huggingface_hub \
  numpy tqdm scipy scikit-learn \
  nvidia-cuda-nvrtc-cu12

# The ABI suffix must match how torch itself was built, or the extension fails to load with an
# undefined-symbol error at import rather than at install.
FA_ABI=$(python -c "import torch; print('TRUE' if torch._C._GLIBCXX_USE_CXX11_ABI else 'FALSE')")
FA_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v$FLASH_ATTN_VERSION/flash_attn-$FLASH_ATTN_VERSION+cu12torch2.8cxx11abi$FA_ABI-cp311-cp311-linux_x86_64.whl"
echo "[build-env] flash-attn $FLASH_ATTN_VERSION (cxx11abi$FA_ABI)"
uv pip install "$FA_WHEEL" || echo "[build-env] WARNING: flash-attn install failed; the eval will fall back to sdpa and may OOM at long rungs"

# ---- veomni ------------------------------------------------------------------------------------
# --no-deps: its full dependency set pins torch/transformers and would rebuild what we just
# installed, for ~7 trivial symbols the modeling code imports (logging, parallel state, a
# checkpointing base class). The cost of --no-deps is that `import veomni` then fails on whatever
# its package __init__ happens to touch, so resolve those one at a time and SAY which ones were
# added -- a hand-maintained list here would silently rot against the pinned ref.
echo "[build-env] veomni (--no-deps) @ $VEOMNI_REF"
uv pip install --no-deps "git+https://github.com/ByteDance-Seed/VeOmni.git@$VEOMNI_REF"
VEOMNI_EXTRA=""
for _ in $(seq 1 15); do
  # The marker is not decoration: veomni prints an INFO banner ("VeOmni ops patch applied") to
  # STDOUT on a SUCCESSFUL import, so a probe that just reads stdout treats that banner as the
  # name of a missing module and tries to pip-install it.
  missing=$(python - <<'PY' | sed -n 's/^__MISSINGMOD__://p' | tail -1
try:
    import veomni  # noqa: F401
except ModuleNotFoundError as e:
    print("__MISSINGMOD__:" + (e.name or ""))
except Exception:
    pass  # a non-import error means the module tree is present; stop resolving
PY
)
  [ -z "$missing" ] && break
  echo "[build-env]   veomni needs '$missing' -- installing"
  uv pip install "$missing" || { echo "[build-env]   FATAL: cannot install '$missing'"; exit 1; }
  VEOMNI_EXTRA="$VEOMNI_EXTRA $missing"
done
python -c "import veomni" || { echo "[build-env] FATAL: veomni still not importable"; exit 1; }
echo "[build-env] veomni OK (extra deps added:${VEOMNI_EXTRA:- none})"
fi  # TILELANG_ONLY

# ---- tilelang: pick the first version whose kernels actually compile ----------------------------
[ -d "$HILS_REPO/models/FlashHiLS" ] || git clone --quiet "$HILS_GIT" "$HILS_REPO"
cat > /tmp/tl_probe.py <<'PY'
"""Compile+run the HiLS chunk-pool kernel on tiny tensors. Import alone is not evidence: a wheel
can import fine and then fail to build a kernel (missing nvcc, wrong tvm-ffi, arch mismatch), and
that failure would otherwise surface in the first eval job instead of here."""
import os, sys, torch
sys.path.insert(0, os.environ["HILS_REPO"])
import tilelang  # noqa: F401
from ops.chunk_attn_pool_tilelang import chunk_attn_pool_tilelang

B, N, S, H, D = 1, 4, 64, 4, 128          # chunk_size 64 and head_dim 128 match the released 7B
mu_q = torch.randn(B, N, H, D, dtype=torch.bfloat16, device="cuda")
k_chunked = torch.randn(B, N, S, H, D, dtype=torch.bfloat16, device="cuda")
lmk_k, lmk_b = chunk_attn_pool_tilelang(mu_q, k_chunked, 1.0 / (D ** 0.5))
assert lmk_k.shape == (B, N, H, D), lmk_k.shape
assert torch.isfinite(lmk_k.float()).all() and torch.isfinite(lmk_b.float()).all()
print(f"    kernel OK: lmk_k{tuple(lmk_k.shape)} lmk_b{tuple(lmk_b.shape)} tilelang={tilelang.__version__}")
PY

export HILS_REPO

# ---- CUDA 12 toolchain for tilelang's JIT ------------------------------------------------------
# There is no nvcc in this image and no pip route to a CUDA 12 one (tilelang's env.py: "only
# nvidia-cuda-nvcc>=13.0 works. nvidia-cuda-nvcc-cu12, etc. only installs `ptxas`, not `nvcc`", and
# the 12.9.86 wheel does indeed ship ptxas alone). Taking the >=13 route would put a CUDA 13
# compiler in front of a CUDA 12.8 stack (torch 2.8.0+cu128, driver 570), so unpack NVIDIA's own
# CUDA 12.8 redist instead: no solver, no channel, deterministic, and it lands on weka next to the
# venv so every eval job gets the identical compiler.
# cuda_cudart comes along for the headers -- nvcc alone cannot compile a kernel that includes
# cuda_runtime.h.
CUDA12_PREFIX="${CUDA12_PREFIX:-$ENV_ROOT/cuda12}"
export CUDA12_PREFIX
# Check the HEADERS too, not just the binary. A partially-installed prefix is the likely state
# after any failed run -- the components are unpacked one at a time, so an earlier failure on
# cuda_cudart leaves a prefix that HAS bin/nvcc and passes a nvcc-only guard, while every kernel
# compile then dies on `fatal error: cuda_runtime.h: No such file or directory`. That exact
# sequence produced four bogus "tilelang rejected" verdicts (jobs ...gpu5 / ...gpu6).
# ONE list, used both to decide whether to (re)install and to verify afterwards. When these two
# checks disagreed, a prefix missing only the newly-required header passed the trigger (so no
# reinstall) and then failed the verification -- an unfixable-looking FATAL on every rerun.
CUDA_REQUIRED_FILES="bin/nvcc include/cuda_runtime.h include/nv/target"
_cuda_ok=1
for f in $CUDA_REQUIRED_FILES; do [ -e "$CUDA12_PREFIX/$f" ] || _cuda_ok=0; done
if [ "$_cuda_ok" != "1" ]; then
  rm -rf "$CUDA12_PREFIX"
  echo "[build-env] installing the CUDA $CUDA_REDIST_MANIFEST toolchain -> $CUDA12_PREFIX"
  mkdir -p "$CUDA12_PREFIX" /tmp/cuda_redist
  # Read the component paths out of NVIDIA's manifest rather than composing them. Each component
  # carries its OWN patch version -- in 12.8.1, cuda_nvcc is 12.8.93 while cuda_cudart is 12.8.90 --
  # so a single hardcoded version 404s for one of them, and `curl -sL` then hands tar an HTML error
  # page ("tar: Error is not recoverable").
  MANIFEST="https://developer.download.nvidia.com/compute/cuda/redist/redistrib_$CUDA_REDIST_MANIFEST.json"
  # Download to a FILE rather than piping into `python - <<'PY'`: the heredoc becomes python's
  # stdin, so json.load(sys.stdin) reads the exhausted heredoc, not the manifest, and every run
  # fails with "could not read".
  curl -fsSL "$MANIFEST" -o /tmp/cuda_redist/manifest.json || { echo "    FATAL: could not fetch $MANIFEST"; exit 1; }
  urls=$(python - /tmp/cuda_redist/manifest.json <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
base = "https://developer.download.nvidia.com/compute/cuda/redist/"
# The minimal set to compile a cutlass/tilelang kernel, each added because its absence was an
# actual compile failure here:
#   cuda_nvcc   -- the compiler
#   cuda_cudart -- cuda_runtime.h / cuda_fp16.h  ("fatal error: cuda_runtime.h")
#   cuda_cccl   -- libcu++: nv/target, cuda/std/* ("fatal error: nv/target")
for comp in ("cuda_nvcc", "cuda_cudart", "cuda_cccl"):
    print(base + d[comp]["linux-x86_64"]["relative_path"])
PY
) || { echo "    FATAL: could not parse $MANIFEST"; exit 1; }
  [ -n "$urls" ] || { echo "    FATAL: manifest yielded no component URLs"; exit 1; }
  for url in $urls; do
    echo "    $(basename "$url")"
    curl -fsSL "$url" -o /tmp/cuda_redist/comp.tar.xz || { echo "    FATAL: download failed: $url"; exit 1; }
    tar -xf /tmp/cuda_redist/comp.tar.xz -C /tmp/cuda_redist
  done
  # The archives unpack to <comp>-linux-x86_64-<ver>-archive/{bin,include,lib}; merge them into one
  # prefix so CUDA_HOME/bin and CUDA_HOME/include are what nvcc expects.
  for d in /tmp/cuda_redist/*-archive; do cp -a "$d"/. "$CUDA12_PREFIX"/; done
  rm -rf /tmp/cuda_redist
fi
for f in $CUDA_REQUIRED_FILES; do
  [ -e "$CUDA12_PREFIX/$f" ] || { echo "[build-env] FATAL: $CUDA12_PREFIX/$f missing after install"; exit 1; }
done

# shellcheck disable=SC1091
. "$(dirname "${BASH_SOURCE[0]}")/hils_cuda_paths.sh"
HAVE_GPU=$(python -c "import torch;print(1 if torch.cuda.is_available() else 0)")
TILELANG_CHOSEN=""
TVM_FFI_CHOSEN=""
for v in $TILELANG_CANDIDATES; do
  # Two attempts per version: whatever the resolver picks, then with the tvm-ffi pin. Which is
  # right is version-dependent -- 0.1.9's declared apache-tvm-ffi range does not exist on PyPI, so
  # it needs the pin, while a release whose range IS satisfiable is better left alone.
  for ffi in resolved "$TVM_FFI_VERSION"; do
    echo "[build-env] trying tilelang==$v (apache-tvm-ffi: $ffi)"
    uv pip uninstall tilelang apache-tvm-ffi >/dev/null 2>&1 || true
    if ! uv pip install "tilelang==$v" >/dev/null 2>&1; then echo "    install failed"; continue; fi
    if [ "$ffi" != "resolved" ]; then uv pip install "apache-tvm-ffi==$ffi" >/dev/null 2>&1 || true; fi
    if [ "$HAVE_GPU" = "1" ]; then
      # Output to a file, not a pipe: `python probe | tail` would report TAIL's exit status, so
      # every candidate would look like it passed.
      if python /tmp/tl_probe.py >/tmp/tl_probe.log 2>&1; then
        cat /tmp/tl_probe.log
        TILELANG_CHOSEN="$v"; TVM_FFI_CHOSEN="$ffi"
        break
      fi
      echo "    rejected. Last 12 lines:"; tail -12 /tmp/tl_probe.log | sed 's/^/      /'
    else
      # CPU node: the best we can do is prove it imports. The kernel gate then falls to the smoke
      # test -- run this script on a GPU node to close that gap here instead.
      if python -c "import tilelang; print('    imports OK', tilelang.__version__)"; then
        TILELANG_CHOSEN="$v"; TVM_FFI_CHOSEN="$ffi"
        break
      fi
      echo "    rejected (import)"
    fi
  done
  [ -n "$TILELANG_CHOSEN" ] && break
done
if [ -z "$TILELANG_CHOSEN" ]; then
  echo "[build-env] FATAL: no tilelang candidate worked ($TILELANG_CANDIDATES)"
  exit 1
fi
echo "[build-env] tilelang==$TILELANG_CHOSEN (apache-tvm-ffi: $TVM_FFI_CHOSEN)"
[ "$HAVE_GPU" = "1" ] || echo "[build-env] ⚠ no GPU here: tilelang was import-checked only, NOT kernel-checked."

# ---- the real import: HiLS's modeling module -----------------------------------------------------
# `import veomni` above is a much weaker check than it looks. The HiLS modeling code imports veomni
# SUBMODULES, which pull in packages the top-level package does not (diffusers, found this way after
# a "successful" build produced a smoke test that could not load the model at all --
# job 01KZY8WDNY195H9BG3YH32AADR). Exercise the actual chain instead, and resolve what it asks for.
echo "[build-env] importing the HiLS modeling code"
HILS_EXTRA=""
for _ in $(seq 1 15); do
  missing=$(python - <<'PY' | sed -n 's/^__MISSINGMOD__://p' | tail -1
import os, sys
sys.path.insert(0, os.environ["HILS_REPO"])
try:
    import models.FlashHiLS.modeling_olmo_hils  # noqa: F401
except ModuleNotFoundError as e:
    print("__MISSINGMOD__:" + (e.name or ""))
except Exception:
    pass  # not an import problem; let the smoke test surface it
PY
)
  [ -z "$missing" ] && break
  case "$missing" in
    models|ops|utils|data|tasks)
      echo "[build-env] FATAL: '$missing' is a HiLS-repo module, not a package -- \$HILS_REPO is wrong"; exit 1 ;;
  esac
  echo "[build-env]   HiLS modeling needs '$missing' -- installing"
  uv pip install "$missing" || { echo "[build-env]   FATAL: cannot install '$missing'"; exit 1; }
  HILS_EXTRA="$HILS_EXTRA $missing"
done
python - <<'PY' || { echo "[build-env] FATAL: HiLS modeling code still not importable"; exit 1; }
import os, sys
sys.path.insert(0, os.environ["HILS_REPO"])
from models.FlashHiLS.modeling_olmo_hils import HiLSForCausalLM  # noqa: F401
from models.FlashHiLS.configuration_hils import HiLSConfig  # noqa: F401
print("    HiLSForCausalLM importable")
PY
echo "[build-env] HiLS modeling OK (extra deps added:${HILS_EXTRA:- none})"

# ---- record what was built ---------------------------------------------------------------------
# The modeling code and its kernels are part of the measurement, so the env has to be able to say
# what it is when a result is questioned months later.
python - "$HILS_ENV/BUILD_INFO.txt" <<'PY'
import importlib, json, platform, sys
info = {"python": platform.python_version()}
for mod in ("torch", "transformers", "tilelang", "veomni", "einops", "numpy"):
    try:
        info[mod] = getattr(importlib.import_module(mod), "__version__", "ok")
    except Exception as e:
        info[mod] = f"MISSING ({type(e).__name__})"
open(sys.argv[1], "w").write(json.dumps(info, indent=2) + "\n")
print("\n[build-env] BUILD_INFO:")
print(json.dumps(info, indent=2))
PY

echo
echo "[build-env] DONE -- tilelang==$TILELANG_CHOSEN. Activate with: source $HILS_ENV/bin/activate"
