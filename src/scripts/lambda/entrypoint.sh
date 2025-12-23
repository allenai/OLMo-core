#!/bin/bash

USERNAME=$1
shift
WORKSPACE=$1
shift

REPO_DIR=/data/ai2/$USERNAME/OLMo-core
VENV_DIR=/data/ai2/uv/OLMo-core-$USERNAME
DATA_DIR=/data/caia-mltrain/data/

function path_prepend {
  for ((i=$#; i>0; i--)); do
      ARG=${!i}
      if [[ -d "$ARG" ]] && [[ ":$PATH:" != *":$ARG:"* ]]; then
          export PATH="$ARG${PATH:+":$PATH"}"
      fi
  done
}

function set_env_var_from_beaker {
    VAR_NAME=$1
    SECRET_NAME=$2
    SECRET_VALUE=$(beaker secret read --workspace="$WORKSPACE" "$SECRET_NAME") || exit 1
    export "$VAR_NAME"="$SECRET_VALUE"
}

echo "============= Starting setup ============="

path_prepend /data/ai2/bin/

# Debugging info.
echo "PATH: $PATH"
echo "HOME: $HOME"
echo "Using repo dir: $REPO_DIR"
echo "Using venv dir: $VENV_DIR"
echo "Using data dir: $DATA_DIR"
echo "Using Beaker workspace: $WORKSPACE"
echo "Authenticated to Beaker as $(beaker account whoami --format=json | jq -r '.[].name')"

# Change to repo directory.
cd "$REPO_DIR" || exit 1

# Set necessary environment variables.
echo "Setting environment variables..."
export GOOGLE_APPLICATION_CREDENTIALS=/data/ai2/google/credentials.json
export OLMO_SHARED_FS=1

export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT
MASTER_PORT=12345

set_env_var_from_beaker WANDB_API_KEY "${USERNAME}_WANDB_API_KEY" || exit 1

# Activate Python virtual env.
echo "Activating Python virtual environment..."
source "$VENV_DIR/bin/activate"
uv pip freeze

echo "============= Setup complete ============="

set -x
exec torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node="$SLURM_GPUS_ON_NODE" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --node_rank="$SLURM_NODEID" \
    "$@"
