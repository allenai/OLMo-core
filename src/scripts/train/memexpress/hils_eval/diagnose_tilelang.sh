#!/bin/bash
# Diagnose the tilelang import/kernel failure in the HiLS runtime, in ONE GPU job.
#
# Symptom (build-hils-env-gpu2, 2026-08-13): every wheel from 0.1.7.post3 through 0.1.9 dies on
#   ValueError: Type 'ffi.Tensor' already has a registered class (<class 'tvm_ffi.core.Tensor'>);
#   re-registering it with the larger wrapper class <class 'tvm.runtime._tensor.Tensor'> is not
#   supported.
# i.e. the standalone `apache-tvm-ffi` package and the `tvm` bundled in the tilelang wheel both
# claim the same FFI type. tilelang 0.1.9 declares apache-tvm-ffi<0.1.13,>=0.1.11, and PyPI has NO
# 0.1.11 or 0.1.12 (it jumps 0.1.9 -> 0.1.13.post3), so whatever got installed is outside the range
# its author tested. 0.1.6.post2 fails differently ("No CUDA or HIP or MPS available on this
# system") -- that one IMPORTS, then cannot see the GPU, which is a separate and possibly easier
# problem.
#
# Rather than burn one job per hypothesis, try them all and print a table. Strategies:
#   A  uninstall apache-tvm-ffi, let the wheel's bundled tvm own the registration
#   B  pin apache-tvm-ffi==0.1.9 (highest below the declared floor -- the declared range is empty)
#   C  import tvm BEFORE tilelang, so the "larger wrapper" registers first, as the error suggests
#   D  0.1.6.post2 + report what its device detection actually sees
set -uo pipefail

ENV_ROOT="${ENV_ROOT:-/weka/oe-training-default/amandab/envs}"
HILS_ENV="${HILS_ENV:-$ENV_ROOT/hils-py311}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$ENV_ROOT/pythons}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ENV_ROOT/uv-cache}"
HILS_REPO="${HILS_REPO:-/tmp/HiLS-Attention}"
export HILS_REPO

command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
# shellcheck disable=SC1091
source "$HILS_ENV/bin/activate"
[ -d "$HILS_REPO/models/FlashHiLS" ] || git clone --quiet https://github.com/abertsch72/HiLS-Attention.git "$HILS_REPO"

echo "=== environment ==="
python -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
nvidia-smi -L 2>/dev/null | head -4
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
which nvcc 2>/dev/null && nvcc --version 2>/dev/null | tail -2 || echo "nvcc: NOT on PATH"

cat > /tmp/probe.py <<'PY'
"""Import tilelang (optionally importing tvm first) and JIT the HiLS chunk-pool kernel."""
import os, sys, traceback
if os.environ.get("IMPORT_TVM_FIRST") == "1":
    import tvm  # noqa: F401
    print("    imported tvm first:", tvm.__version__ if hasattr(tvm, "__version__") else "ok")
import torch
sys.path.insert(0, os.environ["HILS_REPO"])
try:
    import tilelang
    print("    import tilelang OK:", getattr(tilelang, "__version__", "?"))
except Exception:
    print("    IMPORT FAILED:"); traceback.print_exc(limit=3); raise SystemExit(10)
try:
    from ops.chunk_attn_pool_tilelang import chunk_attn_pool_tilelang
    B, N, S, H, D = 1, 4, 64, 4, 128
    mu_q = torch.randn(B, N, H, D, dtype=torch.bfloat16, device="cuda")
    k_ch = torch.randn(B, N, S, H, D, dtype=torch.bfloat16, device="cuda")
    lmk_k, lmk_b = chunk_attn_pool_tilelang(mu_q, k_ch, 1.0 / (D ** 0.5))
    assert torch.isfinite(lmk_k.float()).all()
    print(f"    KERNEL OK lmk_k{tuple(lmk_k.shape)}")
except Exception:
    print("    KERNEL FAILED:"); traceback.print_exc(limit=4); raise SystemExit(11)
PY

report() { echo; echo "########## $1"; }

report "A: tilelang==0.1.9, apache-tvm-ffi UNINSTALLED"
uv pip uninstall tilelang apache-tvm-ffi >/dev/null 2>&1
uv pip install "tilelang[nvcc]==0.1.9" >/dev/null 2>&1
uv pip uninstall apache-tvm-ffi >/dev/null 2>&1
python /tmp/probe.py; echo "  -> exit $?"

report "B: tilelang==0.1.9 + apache-tvm-ffi==0.1.9 (declared range is empty on PyPI)"
uv pip uninstall tilelang apache-tvm-ffi >/dev/null 2>&1
uv pip install "tilelang[nvcc]==0.1.9" >/dev/null 2>&1
uv pip install "apache-tvm-ffi==0.1.9" >/dev/null 2>&1
python /tmp/probe.py; echo "  -> exit $?"

report "C: tilelang==0.1.9, default deps, but import tvm FIRST"
uv pip uninstall tilelang apache-tvm-ffi >/dev/null 2>&1
uv pip install "tilelang[nvcc]==0.1.9" >/dev/null 2>&1
IMPORT_TVM_FIRST=1 python /tmp/probe.py; echo "  -> exit $?"

report "D: tilelang==0.1.6.post2 (imports, but reported no CUDA)"
uv pip uninstall tilelang apache-tvm-ffi >/dev/null 2>&1
uv pip install "tilelang[nvcc]==0.1.6.post2" >/dev/null 2>&1
python /tmp/probe.py; echo "  -> exit $?"
python - <<'PY'
try:
    import tilelang, torch
    from tilelang.utils.target import determine_target
    print("    determine_target ->", determine_target(return_object=False))
except Exception as e:
    print("    determine_target failed:", type(e).__name__, e)
PY

echo; echo "=== what is installed now ==="
uv pip list 2>/dev/null | grep -iE "tilelang|tvm|torch " || true
