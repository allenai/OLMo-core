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

function node_0_only {
    if [ "$SLURM_NODEID" -eq 0 ]; then
        "$@"
    fi
}

node_0_only echo "============= Starting setup ============="

path_prepend /data/ai2/bin/

# Debugging info.
node_0_only echo "HOSTNAME: $(hostname)"
node_0_only echo "PATH: $PATH"
node_0_only echo "HOME: $HOME"
node_0_only echo "Using repo dir: $REPO_DIR"
node_0_only echo "Using venv dir: $VENV_DIR"
node_0_only echo "Using data dir: $DATA_DIR"

# Change to repo directory.
cd "$REPO_DIR" || exit 1

# Set necessary environment variables.
node_0_only echo "Setting environment variables..."
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
node_0_only echo "MASTER_ADDR: $MASTER_ADDR"
node_0_only echo "MASTER_PORT: $MASTER_PORT"

# Ensure port is available.
# if ! nc -vz "$MASTER_ADDR" $MASTER_PORT; then
#     echo "Error: Master port $MASTER_PORT on $MASTER_ADDR is not available."
#     exit 1
# fi

# Activate Python virtual env.
node_0_only echo "Activating Python virtual environment..."
source "$VENV_DIR/bin/activate"
node_0_only uv pip freeze

# Run per-node healthchecks (fails fast on bad nodes).
node_0_only echo "Running per-node healthchecks..."
if ! ./src/scripts/lambda/healthchecks.sh; then
    echo "Healthcheck failed on node '$(hostname)'. Consider adding it to the cordoned list by running:"
    echo ""
    echo "  echo $(hostname) >> /data/ai2/cordoned-nodes.txt"
    echo ""
    exit 1
fi

node_0_only echo "============= Setup complete ============="

exec torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node="$SLURM_GPUS_ON_NODE" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --node_rank="$SLURM_NODEID" \
    "$@"
