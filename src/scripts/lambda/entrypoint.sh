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

path_prepend /data/ai2/bin/

echo "============= Starting setup ============="

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
source "$VENV_DIR/bin/activate"
uv pip freeze

# Install necessary Beaker secrets.
# beaker secret read --workspace="$WORKSPACE" GOOGLE_CREDENTIALS

echo "============= Setup complete ============="

exec "$@"
