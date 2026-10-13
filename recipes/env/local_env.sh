#!/bin/bash
# Common environment for ALL local (Berkeley slurm) OLMo-core jobs — train, eval, data.
# Source this right after the #SBATCH header instead of copy-pasting the conventions:
#
#   source <repo>/recipes/env/local_env.sh   (absolute path; sbatch runs a spooled copy of the launcher)
#
# What it does (see local_cluster.md for the why):
#   - REPO           = this repo's root (derived from this file's location)
#   - env on PATH    = fast /data clone of corpus-reasoning-olmo; direct PATH, never
#                      `conda activate` (activate hangs minutes on NFS lock contention). NO NFS
#                      fallback: a node without the /data env fails here, loudly.
#   - PY             = that env's python. Launch multi-GPU work as
#                      `"$PY" -m torch.distributed.run ...`, NEVER bare `torchrun` (it can resolve
#                      to the NFS miniforge and park every rank in D state)
#   - PYTHONPATH     = $REPO/src (redundant with the editable install, kept as a safety net)
#   - offline flags  = HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE=1 (compute-node egress is blocked/slow;
#                      cache-only avoids silent multi-minute hub-retry stalls). The cache must be
#                      warm — for a NEW model, download once from the login node (has egress), or
#                      pre-set HF_HUB_OFFLINE=0 before sourcing to override.
#   - PYTHONWARNINGS = ignore (8 ranks' FutureWarning storm into one log re-creates the NFS stall)
#   - WANDB_API_KEY  from ~/.netrc if unset; WANDB_FLAG="--no-wandb" when no creds
#   - MASTER_PORT    randomized for torch.distributed.run (avoids collisions between co-located jobs)
# Functions:
#   - fresh_workdir ROOT  -> echoes a fresh per-job work dir ROOT/cache-$SLURM_JOB_ID (stale NFS
#                            locks from killed jobs poison reused work dirs)
#
# NOT covered (slurm parses these before the script runs — keep them in each launcher's header):
#   #SBATCH --output=/data/prasann/joblogs/<name>_%j.log   <- NEVER /accounts or /scratch.
#       slurm opens it BEFORE the script runs and does not create the dir: if the target node has
#       no /data/prasann/joblogs the job dies with no log at all. Create it once per node:
#         srun -p <partition> -w <node> mkdir -p /data/prasann/joblogs
#   partition/qos/account/nodelist/gres                    <- see local_cluster.md cluster map
#
# Hung job? (see local_cluster.md)
#   - Python trainers that call recipes/train/run_guards.install_stack_dump_handler() print every
#     thread's stack to their log on `kill -USR1 <pid>` (without killing the rank).
#   - py-spy is OPTIONAL and not in the env by default; install it once per node-local env with
#       /data/prasann/conda/envs/corpus-reasoning-olmo/bin/python -m pip install py-spy
#     then `py-spy dump --pid <pid>` (also used by `ops stall` / `ops guard` when present).

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export REPO

_ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
if [ ! -x "$_ENV/bin/python" ]; then
  echo "local_env: FATAL: no node-local env at $_ENV on $(hostname -s); an NFS interpreter deadlocks in D state" >&2
  if [ -n "${SLURM_JOB_ID:-}" ]; then exit 1; else return 1; fi
fi
export PATH="$_ENV/bin:$PATH"
export PY="$_ENV/bin/python"

export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
: "${HF_HUB_OFFLINE:=1}"; export HF_HUB_OFFLINE
: "${TRANSFORMERS_OFFLINE:=1}"; export TRANSFORMERS_OFFLINE
: "${PYTHONWARNINGS:=ignore}"; export PYTHONWARNINGS

if [ -z "${WANDB_API_KEY:-}" ]; then
  WANDB_API_KEY=$(awk '/machine api.wandb.ai/{f=1} f&&/password/{print $2; exit}' "$HOME/.netrc" 2>/dev/null)
  export WANDB_API_KEY
fi
WANDB_FLAG=""
[ -z "${WANDB_API_KEY:-}" ] && WANDB_FLAG="--no-wandb" && echo "local_env: no wandb creds -> WANDB_FLAG=--no-wandb"

export MASTER_PORT=$((29000 + RANDOM % 1000))

fresh_workdir() {
  local root="${1:?usage: fresh_workdir ROOT}"
  local d="$root/cache-${SLURM_JOB_ID:-$$}"
  mkdir -p "$d"
  echo "$d"
}
