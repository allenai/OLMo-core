#!/bin/bash
# Common environment for ALL local (Berkeley slurm) OLMo-core jobs — train, eval, data.
# Source this right after the #SBATCH header instead of copy-pasting the conventions:
#
#   source /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/src/scripts/local_env.sh
#
# What it does (see local_cluster.md for the why):
#   - REPO           = this repo's root (derived from this file's location)
#   - env on PATH    = fast /data clone of corpus-reasoning-olmo, NFS fallback; direct PATH,
#                      never `conda activate` (activate hangs minutes on NFS lock contention)
#   - PYTHONPATH     = $REPO/src (redundant with the editable install, kept as a safety net)
#   - offline flags  = HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE=1 (compute-node egress is blocked/slow;
#                      cache-only avoids silent multi-minute hub-retry stalls). The cache must be
#                      warm — for a NEW model, download once from the login node (has egress), or
#                      pre-set HF_HUB_OFFLINE=0 before sourcing to override.
#   - PYTHONWARNINGS = ignore (8 ranks' FutureWarning storm into one log re-creates the NFS stall)
#   - WANDB_API_KEY  from ~/.netrc if unset; WANDB_FLAG="--no-wandb" when no creds
#   - MASTER_PORT    randomized for torchrun (avoids collisions between co-located jobs)
# Functions:
#   - pick_clean_gpus N   -> CUDA_VISIBLE_DEVICES = first N GPUs with <2GB used (allocations
#                            regularly include a rogue-occupied GPU; nvidia-smi isn't cgroup-isolated)
#   - fresh_workdir ROOT  -> echoes a fresh per-job work dir ROOT/cache-$SLURM_JOB_ID (stale NFS
#                            locks from killed jobs poison reused work dirs)
#
# NOT covered (slurm parses these before the script runs — keep them in each launcher's header):
#   #SBATCH --output=/data/prasann/joblogs/<name>_%j.log   <- NEVER /accounts or /scratch
#   partition/qos/account/nodelist/gres                    <- see local_cluster.md cluster map

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export REPO

_ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$_ENV" ] || _ENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
export PATH="$_ENV/bin:$PATH"

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

pick_clean_gpus() {
  local n="${1:?usage: pick_clean_gpus N}"
  local clean
  mapfile -t clean < <(nvidia-smi --query-gpu=memory.used,uuid --format=csv,noheader,nounits |
    awk -F', ' '$1+0 < 2000 {print $2}')
  if [ "${#clean[@]}" -lt "$n" ]; then
    echo "local_env: WARNING only ${#clean[@]} clean GPUs for requested $n" >&2
  fi
  CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${clean[*]:0:$n}")
  export CUDA_VISIBLE_DEVICES
  echo "local_env: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
}

fresh_workdir() {
  local root="${1:?usage: fresh_workdir ROOT}"
  local d="$root/cache-${SLURM_JOB_ID:-$$}"
  mkdir -p "$d"
  echo "$d"
}
