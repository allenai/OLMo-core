#!/bin/bash
# Activate the HiLS runtime and check out the HiLS repo.
#
# Sourced (not executed) by the on-node eval runner and the smoke test, because it EXPORTS
# $HILS_REPO and switches the active python:  `source hils_env_setup.sh`
#
# The runtime itself is a Python 3.11 venv built ONCE on weka by build_hils_env_weka.sh. It is not
# built here because the OLMo-core image's own python is 3.12, on which the HiLS stack does not
# run at all:
#   * veomni declares requires-python '<3.12,>=3.11'
#   * tilelang 0.1.9 dies on import under 3.12 (AttributeError: attribute '__dict__' of 'type'
#     objects is not writable)
# Both were observed on beaker (job 01KZY56CDDR39TCHXDF65F4EY1, 2026-08-13) before this split.
set -uo pipefail

ENV_ROOT="${ENV_ROOT:-/weka/oe-training-default/amandab/envs}"
HILS_ENV="${HILS_ENV:-$ENV_ROOT/hils-py311}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$ENV_ROOT/pythons}"

# Pinned to the commit the released 7B checkpoint was published against. A floating `main` would
# make an eval silently non-reproducible -- the modeling code IS part of the measurement here.
HILS_REPO="${HILS_REPO:-/tmp/HiLS-Attention}"
HILS_GIT="${HILS_GIT:-https://github.com/abertsch72/HiLS-Attention.git}"
HILS_COMMIT="${HILS_COMMIT:-}"

# ---- runtime ------------------------------------------------------------------------------
if [ ! -f "$HILS_ENV/bin/activate" ]; then
  echo "[hils-env] FATAL: no HiLS runtime at $HILS_ENV."
  echo "           Build it once with build_hils_env_weka.sh (see hils_eval/README.md)."
  return 1 2>/dev/null || exit 1
fi
# shellcheck disable=SC1091
source "$HILS_ENV/bin/activate"
echo "[hils-env] activated $HILS_ENV -- python=$(python -V 2>&1) at $(which python)"
# tilelang JIT-compiles its kernels at runtime, so nvcc/nvrtc must be reachable in EVERY job, not
# just at build time. Without this the failure is a misleading "No CUDA available" on a healthy H100.
# shellcheck disable=SC1091
. "$(dirname "${BASH_SOURCE[0]}")/hils_cuda_paths.sh"

# ---- repo ---------------------------------------------------------------------------------
# HILS_NEED_REPO=0 for a non-HiLS model (the Olmo-3 base control). Those still activate the venv
# above -- running the control in the image's env and the treatment in this one would make the
# comparison span two torch versions for no reason -- but they have no use for the modeling code.
if [ "${HILS_NEED_REPO:-1}" != "1" ]; then
  echo "[hils-env] HILS_NEED_REPO=0 -- runtime only, skipping the HiLS repo checkout."
  return 0 2>/dev/null || exit 0
fi
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

echo "[hils-env] versions:"
python - <<'PY'
import importlib
for mod in ("torch", "transformers", "tilelang", "veomni", "einops", "flash_attn"):
    try:
        m = importlib.import_module(mod)
        print(f"    {mod:14s} {getattr(m, '__version__', 'ok')}")
    except Exception as e:
        print(f"    {mod:14s} MISSING ({type(e).__name__}: {e})")
PY
