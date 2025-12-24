#!/bin/bash

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

function echo_node_0 {
    if [ "$SLURM_NODEID" -eq 0 ]; then
        echo "$@"
    fi
}

echo_node_0 "============= Starting setup ============="

path_prepend /data/ai2/bin/

# Debugging info.
echo_node_0 "HOSTNAME: $(hostname)"
echo_node_0 "PATH: $PATH"
echo_node_0 "HOME: $HOME"
echo_node_0 "Using repo dir: $REPO_DIR"
echo_node_0 "Using venv dir: $VENV_DIR"
echo_node_0 "Using data dir: $DATA_DIR"

# Change to repo directory.
cd "$REPO_DIR" || exit 1

# Set necessary environment variables.
echo_node_0 "Setting environment variables..."
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
echo_node_0 "Activating Python virtual environment..."
source "$VENV_DIR/bin/activate"
uv pip freeze

echo_node_0 "============= Setup complete ============="

set -x
exec torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node="$SLURM_GPUS_ON_NODE" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --node_rank="$SLURM_NODEID" \
    "$@"
