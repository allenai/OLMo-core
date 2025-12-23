#!/bin/bash

USERNAME=$1
shift

REPO_DIR=/data/ai2/$USERNAME/OLMo-core
VENV_DIR=/data/ai2/uv/OLMo-core-$USERNAME
DATA_DIR=/data/caia-mltrain/data/

echo "============= Starting setup ============="

# Debugging info.
echo "PATH: $PATH"
echo "Using repo dir: $REPO_DIR"
echo "Using venv dir: $VENV_DIR"
echo "Using data dir: $DATA_DIR"

# Change to repo directory.
cd "$REPO_DIR" || exit 1

# Activate Python virtual env.
source "$VENV_DIR/bin/activate"
uv pip freeze

echo "============= Setup complete ============="

exec "$@"
