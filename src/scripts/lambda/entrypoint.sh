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

# Change to repo directory.
cd "$REPO_DIR" || exit 1

# Activate Python virtual env.
echo "Activating Python virtual environment..."
source "$VENV_DIR/bin/activate"
uv pip freeze

# Set necessary environment variables.
echo "Setting environment variables..."
export GOOGLE_APPLICATION_CREDENTIALS=/data/ai2/google/credentials.json
set_env_var_from_beaker WANDB_API_KEY "${USERNAME}_WANDB_API_KEY" || exit 1

echo "============= Setup complete ============="

exec "$@"
