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
# Newest first. 0.1.9 imports on 3.11 but collides with the separately-installed apache-tvm-ffi
# ("Type 'ffi.Tensor' already has a registered class") because the wheel also bundles its own tvm.
# Rather than guess which release predates that split, try them in order and keep the first that
# COMPILES A REAL KERNEL. The HiLS requirements.txt pins a git sha instead, which is a 20-40 min
# source build needing LLVM -- not worth it if a wheel works.
TILELANG_CANDIDATES="${TILELANG_CANDIDATES:-0.1.9 0.1.8 0.1.7.post3 0.1.6.post2}"
HILS_REPO="${HILS_REPO:-/tmp/HiLS-Attention}"
HILS_GIT="${HILS_GIT:-https://github.com/abertsch72/HiLS-Attention.git}"

mkdir -p "$ENV_ROOT" "$UV_PYTHON_INSTALL_DIR" "$UV_CACHE_DIR"

command -v uv >/dev/null 2>&1 || {
  echo "[build-env] installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
}
echo "[build-env] uv=$(uv --version)"

if [ "${REBUILD:-0}" = "1" ]; then
  echo "[build-env] REBUILD=1 -- removing $HILS_ENV"
  rm -rf "$HILS_ENV"
fi

echo "[build-env] creating $HILS_ENV (python 3.11)"
uv venv --python 3.11 "$HILS_ENV"
# shellcheck disable=SC1091
source "$HILS_ENV/bin/activate"
echo "[build-env] python=$(python -V) at $(which python)"

echo "[build-env] torch $TORCH_VERSION from $TORCH_INDEX"
uv pip install "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX"

echo "[build-env] transformers + our eval deps"
# flash-attn is deliberately NOT installed: there is no wheel for this torch/python pair, so pip
# would compile it for 30+ minutes, and nothing here needs it. The HiLS sparse path is tilelang;
# flash-attn would only serve the interleaved DENSE layers, which run fine on sdpa. That is a
# speed choice, not a semantic one.
uv pip install \
  "transformers==$TRANSFORMERS_VERSION" \
  einops accelerate safetensors huggingface_hub \
  numpy tqdm scipy scikit-learn

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
HAVE_GPU=$(python -c "import torch;print(1 if torch.cuda.is_available() else 0)")
TILELANG_CHOSEN=""
for v in $TILELANG_CANDIDATES; do
  echo "[build-env] trying tilelang==$v"
  uv pip uninstall tilelang apache-tvm-ffi >/dev/null 2>&1 || true
  uv pip install "tilelang[nvcc]==$v" >/dev/null 2>&1 || { echo "    install failed"; continue; }
  if [ "$HAVE_GPU" = "1" ]; then
    if python /tmp/tl_probe.py; then TILELANG_CHOSEN="$v"; break; fi
  else
    # CPU node: the best we can do is prove it imports. The kernel gate then falls to the smoke
    # test -- run this script on a GPU node to close that gap here instead.
    if python -c "import tilelang; print('    imports OK', tilelang.__version__)"; then
      TILELANG_CHOSEN="$v"; break
    fi
  fi
  echo "    tilelang==$v rejected"
done
[ -n "$TILELANG_CHOSEN" ] || { echo "[build-env] FATAL: no tilelang candidate worked ($TILELANG_CANDIDATES)"; exit 1; }
[ "$HAVE_GPU" = "1" ] || echo "[build-env] ⚠ no GPU here: tilelang was import-checked only, NOT kernel-checked."

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
