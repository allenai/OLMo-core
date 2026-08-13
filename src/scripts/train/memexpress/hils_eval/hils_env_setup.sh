#!/bin/bash
# Install the HiLS-Attention runtime into the current python env and check out the HiLS repo.
#
# Sourced (not executed) by the on-node eval runner and the smoke test, because it EXPORTS
# $HILS_REPO for them:  `source hils_env_setup.sh`
#
# What HiLS needs that our image does not already have:
#   * the HiLS repo itself     -- the modeling code is out-of-tree (no auto_map on the HF repo)
#   * tilelang                 -- JIT CUDA kernels for the chunk-pool + sliding-window attention;
#                                 imported at module load AND called on every forward, so it is
#                                 not optional even for pure inference
#   * veomni                   -- ByteDance's training framework. The modeling code imports ~7
#                                 trivial symbols from it (logging, parallel-state, a checkpointing
#                                 base class). Installed with --no-deps ON PURPOSE: its full
#                                 dependency set pins torch/transformers and would rebuild the
#                                 environment underneath us for symbols we do not use.
#   * einops                   -- used directly by hils_attention.py
#
# Idempotent: every step is skipped if already satisfied, so a rerun on a warm node is seconds.
set -uo pipefail

# Pinned to the commit the released 7B checkpoint was published against. A floating `main` would
# make an eval silently non-reproducible -- the modeling code IS part of the measurement here.
HILS_REPO="${HILS_REPO:-/tmp/HiLS-Attention}"
HILS_GIT="${HILS_GIT:-https://github.com/abertsch72/HiLS-Attention.git}"
HILS_COMMIT="${HILS_COMMIT:-}"
# tilelang: the HiLS requirements.txt pins a git sha (source build, needs LLVM + several minutes).
# The PyPI wheels are cp38-abi3 manylinux, so they drop in with no compile. Pin the version so a
# tilelang release cannot change kernel behaviour between two of our eval jobs.
TILELANG_VERSION="${TILELANG_VERSION:-0.1.9}"
VEOMNI_REF="${VEOMNI_REF:-441e1b2483921e9cfe56c8d97541a23cb4b290a8}"

echo "[hils-env] python=$(which python) torch=$(python -c 'import torch;print(torch.__version__)' 2>/dev/null || echo MISSING)"

# ---- repo ---------------------------------------------------------------------------------
if [ ! -d "$HILS_REPO/models/FlashHiLS" ]; then
  echo "[hils-env] cloning $HILS_GIT -> $HILS_REPO"
  rm -rf "$HILS_REPO"
  git clone --quiet "$HILS_GIT" "$HILS_REPO" || { echo "[hils-env] FATAL: clone failed"; return 1 2>/dev/null || exit 1; }
  if [ -n "$HILS_COMMIT" ]; then
    git -C "$HILS_REPO" checkout --quiet "$HILS_COMMIT" || { echo "[hils-env] FATAL: no commit $HILS_COMMIT"; return 1 2>/dev/null || exit 1; }
  fi
fi
echo "[hils-env] HILS_REPO=$HILS_REPO @ $(git -C "$HILS_REPO" rev-parse --short HEAD)"
export HILS_REPO

# ---- python deps --------------------------------------------------------------------------
# tilelang[nvcc] pulls the nvidia-cuda-nvcc wheel: the release image is not a CUDA *devel* image,
# so without it tilelang's JIT has no compiler and dies on the first forward pass.
python -c "import tilelang" 2>/dev/null || {
  echo "[hils-env] installing tilelang==$TILELANG_VERSION"
  pip install --quiet "tilelang[nvcc]==$TILELANG_VERSION"
}
python -c "import veomni" 2>/dev/null || {
  echo "[hils-env] installing veomni (--no-deps) @ $VEOMNI_REF"
  pip install --quiet --no-deps "git+https://github.com/ByteDance-Seed/VeOmni.git@$VEOMNI_REF"
}
python -c "import einops" 2>/dev/null || pip install --quiet einops

echo "[hils-env] versions:"
python - <<'PY'
import importlib
for mod in ("torch", "transformers", "tilelang", "veomni", "einops", "flash_attn"):
    try:
        m = importlib.import_module(mod)
        print(f"    {mod:14s} {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"    {mod:14s} MISSING ({type(e).__name__}: {e})")
PY
