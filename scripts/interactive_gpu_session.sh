#!/usr/bin/env bash
# Launch a Beaker interactive session with GPUs + weka mounted (for dev on the cluster).
#
# Usage:
#   scripts/interactive_gpu_session.sh                 # 1 GPU, remote (SSH from laptop), jupiter, high prio
#   GPUS=2 scripts/interactive_gpu_session.sh          # 2 GPUs
#   LOCAL=1 scripts/interactive_gpu_session.sh         # if you're already ON a beaker node (no --remote)
#   CLUSTER=ai2/saturn-cirrascale scripts/interactive_gpu_session.sh
#   WANDB_SECRET=PRASANNS_WANDB_API_KEY scripts/interactive_gpu_session.sh   # wire wandb key
#
# Inside the session: clone + install the repo, e.g.
#   git clone -b prasann/landmark https://github.com/allenai/OLMo-core.git && cd OLMo-core && pip install -e .
# Data is at /weka/${WEKA}/ai2-llm/checkpoints/prasanns/cr_suite_data
set -euo pipefail

GPUS="${GPUS:-1}"
CLUSTER="${CLUSTER:-ai2/jupiter-cirrascale-2}"
PRIORITY="${PRIORITY:-high}"            # cluster is usually full -> high/urgent to get scheduled
BUDGET="${BUDGET:-ai2/oe-other}"
WORKSPACE="${WORKSPACE:-ai2/flex2}"
WEKA="${WEKA:-oe-training-default}"
# OLMo-core "stable" image (torch 2.9.1 / cu12.8 + GPU deps). Beaker image -> needs beaker:// scheme.
IMAGE="${IMAGE:-beaker://tylerr/olmo-core-tch291cu128-2025-11-25}"
SHM="${SHM:-32GiB}"
NAME="${NAME:-${USER}-dev-gpu}"

ARGS=(
  --gpus "${GPUS}"
  --priority "${PRIORITY}"
  --budget "${BUDGET}"
  --workspace "${WORKSPACE}"
  --image "${IMAGE}"
  --mount "src=weka://${WEKA},dst=/weka/${WEKA}"
  --shared-memory "${SHM}"
  --name "${NAME}"
)

# Remote (SSH into the session from a laptop) requires --bare and --cluster. Set LOCAL=1 if you're
# already running the beaker client on a cluster node (then it runs there, attaches directly).
if [ "${LOCAL:-0}" != "1" ]; then
  ARGS+=(--remote --bare --cluster "${CLUSTER}")
fi

# Optional: wire your wandb key from a workspace secret (set WANDB_SECRET to the secret's name).
if [ -n "${WANDB_SECRET:-}" ]; then
  ARGS+=(--secret-env "WANDB_API_KEY=${WANDB_SECRET}")
fi

echo "+ beaker session create ${ARGS[*]}"
exec beaker session create "${ARGS[@]}"
