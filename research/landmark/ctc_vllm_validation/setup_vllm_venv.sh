#!/bin/bash
# Build an isolated vLLM venv on /data (node-local, fast) with a recent-enough vLLM to support
# Qwen3.5 (hybrid mamba+attention) text-only serving + the FlexAttention v1 backend the
# chunked-mask patch (corpus_reasoning.lib.vllm_chunked_patch) needs.
#
# LOCKED + marker-gated (fixed 2026-07-21): the fan-out runs several prep jobs per node
# concurrently, and each one used to call this script -- with no lock, N concurrent
# `pip install` invocations into the SAME site-packages directory raced and corrupted the venv
# (sneetches: 6 concurrent installs left torch/vllm/safetensors all missing despite pip
# reporting "already satisfied" for everything else). flock serializes; the DONE marker makes
# every call after the first a fast no-op instead of re-running (and re-risking) pip.
set -uo pipefail
VENV=/data/prasann/ctc_vllm_venv
BASE_PY=/scratch/users/prasann/conda/envs/corpus-reasoning-eval/bin/python
LOCK=/data/prasann/.ctc_vllm_venv_setup.lock
MARKER=/data/prasann/.ctc_vllm_venv_setup_done

# Comprehensive node-local preamble (self-healing) -- see gpu_eval_task.sh's 2026-07-21
# root-cause note. HOME alone was not enough; cwd + leaked CONDA_ENVS_PATH also needed forcing
# onto /data (pip/venv creation, and anything it shells out to, can touch either).
PREAMBLE=/data/prasann/node_local_env.sh
if [ ! -f "$PREAMBLE" ]; then
  cat > "$PREAMBLE" <<'ENVEOF'
export HOME=/data/prasann/home
export TMPDIR=/data/prasann/tmp
export XDG_CACHE_HOME=/data/prasann/xdg_cache
export XDG_CONFIG_HOME=/data/prasann/xdg_config
export FLASHINFER_CACHE_DIR=/data/prasann/flashinfer_cache
export CUDA_CACHE_PATH=/data/prasann/cuda_cache
export TRITON_CACHE_DIR=/data/prasann/triton_cache
export TORCHINDUCTOR_CACHE_DIR=/data/prasann/inductor_cache
export HF_HOME=/data/prasann/hf
export TRANSFORMERS_CACHE=/data/prasann/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PIP_CACHE_DIR=/data/prasann/pip_cache
REPO_ROOT=${REPO_ROOT:-$(git -C "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" rev-parse --show-toplevel)}
export PYTHONPATH=$REPO_ROOT/research/common:$REPO_ROOT/src
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:/data/prasann/conda/envs/corpus-reasoning-olmo/bin:/data/prasann/ctc_vllm_venv/bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONWARNINGS=ignore
export CONDA_ENVS_PATH=/data/prasann/conda/envs
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_PROMPT_MODIFIER CONDA_PKGS_DIRS 2>/dev/null
ENVEOF
fi
source "$PREAMBLE"
mkdir -p "$HOME" "$HOME/.cache" "$TMPDIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" \
  "$FLASHINFER_CACHE_DIR" "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
  "$HF_HOME" "$PIP_CACHE_DIR"
# AWS creds: moving HOME to /data broke the ~/.aws lookup (AWS CLI resolves it from
# $HOME, and the real creds live under the NFS home). Copy once, self-healing.
if [ ! -f "$HOME/.aws/credentials" ]; then
  mkdir -p "$HOME/.aws"
  cp /accounts/projects/berkeleynlp/prasann/.aws/config /accounts/projects/berkeleynlp/prasann/.aws/credentials "$HOME/.aws/" 2>/dev/null
  chmod 600 "$HOME/.aws/config" "$HOME/.aws/credentials" 2>/dev/null
fi

exec 9>"$LOCK"
flock -w 900 9 || { echo "!!! [setup_vllm_venv] could not acquire lock within 900s on $(hostname)"; exit 1; }

if [ -f "$MARKER" ] && "$VENV/bin/python" -c "import vllm, torch, safetensors" 2>/dev/null; then
  echo "[setup_vllm_venv] already complete on $(hostname) ($(cat "$MARKER")) -- skipping"
  exit 0
fi

echo "=== HOST=$(hostname) START=$(date '+%F %T') ==="
if [ -x "$VENV/bin/python" ]; then
  echo "venv dir already exists at $VENV (marker missing or import check failed -- re-verifying/reinstalling)"
else
  "$BASE_PY" -m venv "$VENV"
fi
export PIP_CACHE_DIR=/data/prasann/pip_cache
mkdir -p "$PIP_CACHE_DIR"
"$VENV/bin/pip" install --upgrade pip
echo "=== installing vllm (this may take a while) $(date '+%F %T') ==="
"$VENV/bin/pip" install "vllm==0.25.1" 2>&1 | tail -100
rc=${PIPESTATUS[0]}
if [ "$rc" -ne 0 ]; then
  echo "!!! [setup_vllm_venv] pip install vllm FAILED rc=$rc on $(hostname)"
  exit 1
fi
echo "=== vllm install done $(date '+%F %T') ==="
"$VENV/bin/python" -c "
import vllm, transformers
print('vllm', vllm.__version__)
print('transformers', transformers.__version__)
from vllm.model_executor.models.registry import _TEXT_GENERATION_MODELS
hits = [k for k in _TEXT_GENERATION_MODELS if 'qwen3_5' in k.lower() or 'qwen3.5' in k.lower()]
print('qwen3_5 registry hits:', hits)
"
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "!!! [setup_vllm_venv] post-install import check FAILED rc=$rc on $(hostname)"
  exit 1
fi
date '+%F %T' > "$MARKER"
echo "=== DONE $(date '+%F %T') ==="
