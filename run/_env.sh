#!/usr/bin/env bash
# Shared environment resolution for every run/*.sh entry point.
#
# This file exists so the cluster traps are written down ONCE. They were previously re-encoded
# (and periodically forgotten) across ~40 sbatch headers, and each of the exports below stands in
# for a failure that cost real debugging time:
#
#   * /accounts and /scratch are BOTH shared NFS at roughly 5 MB/s. A torch or vLLM process
#     launched from an NFS interpreter pages multiple GB of shared objects over that link and
#     parks in D state at ~0% CPU for minutes. It looks like a hung GPU or a slow model load and
#     is neither. Diagnose with: ps -o stat,wchan,%cpu  ->  Dl + nfs_wait_bit_killabl + ~0% CPU.
#   * The GDN Triton/nvcc JIT leaks ~165 MB per job into TMPDIR; pointed at /tmp it fills the node.
#   * flashinfer caches under $HOME. With HOME on NFS, concurrent jobs wedge each other.
#   * CUDA_HOME must be a REAL system toolkit for the GDN JIT to compile.
#
# Everything here is overridable from the environment, so a different cluster can set its own
# values without editing this file.

set -euo pipefail

# --- node-local scratch -------------------------------------------------------------------------
# The only fast storage is the target node's own disk. Fall back to /tmp on hosts without /data
# (e.g. the login node), where the caches are small and short-lived anyway.
if [[ -d /data ]] && [[ -w /data ]]; then
  CTC_LOCAL="${CTC_LOCAL:-/data/$USER/ctc_run}"
else
  CTC_LOCAL="${CTC_LOCAL:-/tmp/$USER/ctc_run}"
fi
mkdir -p "$CTC_LOCAL"/{tmp,home,cache,flashinfer,triton}

export TMPDIR="${TMPDIR_OVERRIDE:-$CTC_LOCAL/tmp}"
export FLASHINFER_CACHE_DIR="${FLASHINFER_CACHE_DIR:-$CTC_LOCAL/flashinfer}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CTC_LOCAL/triton}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$CTC_LOCAL/cache}"
export HF_HOME="${HF_HOME:-$CTC_LOCAL/cache/huggingface}"

# NOTE: HOME is deliberately NOT overridden by default.
#
# Moving HOME does keep stray ~/.cache writes off NFS, but it also moves Python's user site-packages
# (~/.local/lib/pythonX.Y/site-packages), which silently hides any `pip install --user` or editable
# install -- `import ctc` then fails with ModuleNotFoundError under the wrapper while working fine
# outside it. That cost time once already; the cache variables above target the actual problem
# without the side effect.
#
# If a tool ignores its cache variable and insists on $HOME (flashinfer has done this), opt in per
# job with CTC_ISOLATE_HOME=1 and make sure the interpreter is a real venv, not a user-site install.
if [[ -n "${CTC_ISOLATE_HOME:-}" ]]; then
  export HOME="$CTC_LOCAL/home"
fi

# --- CUDA ---------------------------------------------------------------------------------------
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
if [[ -d "$CUDA_HOME/bin" ]]; then
  export PATH="$CUDA_HOME/bin:$PATH"
fi

# --- interpreter --------------------------------------------------------------------------------
# Prefer an explicitly configured node-local interpreter. NEVER default to one on /accounts or
# /scratch for a torch/vLLM job -- see the NFS note above.
CTC_PYTHON="${CTC_PYTHON:-}"
if [[ -z "$CTC_PYTHON" ]]; then
  for candidate in /data/prasann/ctc_vllm_venv/bin/python "$(command -v python3 || true)"; do
    if [[ -n "$candidate" && -x "$candidate" ]]; then
      CTC_PYTHON="$candidate"
      break
    fi
  done
fi
if [[ -z "$CTC_PYTHON" ]]; then
  echo "run/_env.sh: no python found; set CTC_PYTHON=/path/to/python" >&2
  exit 1
fi
export CTC_PYTHON

case "$CTC_PYTHON" in
  /net/*)
    # /net/<node>/data is the same bytes as that node's /data, reached over NFS. It exists for
    # auditing another host, never for running anything -- and a job on the node it names would be
    # reading its own local disk the slow way.
    echo "run/_env.sh: WARNING: interpreter is a /net path ($CTC_PYTHON)." >&2
    echo "  /net is the NFS view of a node's local disk: read-only auditing, not job I/O." >&2
    echo "  On that node, use the equivalent /data/... path instead." >&2
    ;;
  /accounts/*|/scratch/*)
    echo "run/_env.sh: WARNING: interpreter is on NFS ($CTC_PYTHON)." >&2
    echo "  Fine for --help and data generation; a torch/vLLM job will stall for minutes." >&2
    echo "  Set CTC_PYTHON to a node-local venv for real runs." >&2
    ;;
esac

CTC_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CTC_REPO

if [[ -n "${CTC_VERBOSE:-}" ]]; then
  echo "[ctc] python=$CTC_PYTHON"
  echo "[ctc] repo=$CTC_REPO  local=$CTC_LOCAL"
  echo "[ctc] TMPDIR=$TMPDIR  HOME=$HOME  CUDA_HOME=$CUDA_HOME"
fi
