#!/bin/bash

USERNAME=$1
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

echo "============= Starting setup ============="

path_prepend /data/ai2/bin/

# Debugging info.
echo "HOSTNAME: $(hostname)"
echo "PATH: $PATH"
echo "HOME: $HOME"
echo "Using repo dir: $REPO_DIR"
echo "Using venv dir: $VENV_DIR"
echo "Using data dir: $DATA_DIR"

# Change to repo directory.
cd "$REPO_DIR" || exit 1

# Set necessary environment variables.
echo "Setting environment variables..."
export GOOGLE_APPLICATION_CREDENTIALS=/data/ai2/google/credentials.json
export OLMO_SHARED_FS=1
export OLMO_RICH_LOGGING=1
export OMP_NUM_THREADS=8
export FORCE_COLOR=1
export TORCH_LOGS=recompiles,graph_breaks

# Resolve hostname of master address and port to use.
export MASTER_ADDR
MASTER_ADDR=$(scontrol show hostname "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT
MASTER_PORT=$((60000 + SLURM_JOB_ID % 5000))

# Ensure port is available.
# if ! nc -vz "$MASTER_ADDR" $MASTER_PORT; then
#     echo "Error: Master port $MASTER_PORT on $MASTER_ADDR is not available."
#     exit 1
# fi

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
