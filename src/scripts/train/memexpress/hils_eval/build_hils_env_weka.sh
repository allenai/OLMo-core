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
# GPU node. The env is built once and mounted read-only-ish by every job at a FIXED path, so the
# venv's absolute shebangs resolve identically everywhere.
#
# Run as a CPU gantry job (see hils_eval/README.md), then every eval job just sources
# hils_env_setup.sh, which activates this.
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
TILELANG_VERSION="${TILELANG_VERSION:-0.1.9}"
VEOMNI_REF="${VEOMNI_REF:-441e1b2483921e9cfe56c8d97541a23cb4b290a8}"

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

echo "[build-env] HiLS runtime + our eval deps"
# flash-attn is deliberately NOT installed: there is no wheel for this torch/python pair, so pip
# would compile it for 30+ minutes, and nothing here needs it. The HiLS sparse path is tilelang;
# flash-attn would only serve the interleaved DENSE layers, which run fine on sdpa. That is a
# speed choice, not a semantic one.
uv pip install \
  "transformers==$TRANSFORMERS_VERSION" \
  "tilelang[nvcc]==$TILELANG_VERSION" \
  einops accelerate safetensors huggingface_hub \
  numpy tqdm scipy scikit-learn
# --no-deps: veomni's full dependency set pins torch/transformers and would rebuild what we just
# installed. The HiLS modeling code imports ~7 trivial symbols from it.
# --ignore-requires-python: veomni declares <3.12 and we ARE on 3.11, but uv still resolves the
# git dependency's marker strictly in some configurations; the flag makes the intent explicit.
uv pip install --no-deps "git+https://github.com/ByteDance-Seed/VeOmni.git@$VEOMNI_REF"

echo
echo "[build-env] verifying imports:"
python - <<'PY'
import importlib, sys
bad = []
for mod in ("torch", "transformers", "tilelang", "veomni", "einops", "numpy", "safetensors",
            "tqdm", "scipy", "sklearn"):
    try:
        m = importlib.import_module(mod)
        print(f"    {mod:14s} {getattr(m, '__version__', 'ok')}")
    except Exception as e:
        print(f"    {mod:14s} FAILED ({type(e).__name__}: {e})")
        bad.append(mod)
print(f"\n    python {sys.version}")
if bad:
    raise SystemExit(f"FAILED imports: {bad}")
PY

echo
echo "[build-env] DONE. Activate with: source $HILS_ENV/bin/activate"
