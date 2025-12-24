#!/bin/bash

# Source helper functions.
. ./src/scripts/lambda/utils.sh

REPO_DIR=/data/ai2/$USERNAME/OLMo-core
VENV_DIR=/data/ai2/uv/OLMo-core-$USERNAME
DATA_DIR=/data/caia-mltrain/data/

node_0_only log_info "Starting setup..."

# Debugging info.
node_0_only log_info "PATH: $PATH"
node_0_only log_info "HOME: $HOME"
node_0_only log_info "Using repo dir: $REPO_DIR"
node_0_only log_info "Using venv dir: $VENV_DIR"
node_0_only log_info "Using data dir: $DATA_DIR"

# Change to repo directory.
cd "$REPO_DIR" || exit 1

# Set necessary environment variables.
node_0_only log_info "Setting environment variables..."
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
node_0_only log_info "MASTER_ADDR: $MASTER_ADDR"
node_0_only log_info "MASTER_PORT: $MASTER_PORT"

# Increase open file limit
ulimit -n 65536
node_0_only log_info "Open File Limit (ulimit -n): $(ulimit -n)"

# Ensure port is available.
# if ! nc -vz "$MASTER_ADDR" $MASTER_PORT; then
#     echo "Error: Master port $MASTER_PORT on $MASTER_ADDR is not available."
#     exit 1
# fi

# Activate Python virtual env.
node_0_only log_info "Activating Python virtual environment..."
source "$VENV_DIR/bin/activate"
node_0_only uv pip freeze

# Run per-node healthchecks (fails fast on bad nodes).
node_0_only log_info "Running per-node healthchecks..."
if ! ./src/scripts/lambda/healthchecks.sh; then
    log_error "Healthcheck failed on node '$(hostname)'."
    echo "Consider adding it to the cordoned list by running:"
    echo ""
    echo "  echo $(hostname) >> /data/ai2/cordoned-nodes.txt"
    echo ""
    exit 1
fi

node_0_only log_info "Setup complete."

node_0_only log_info "Launching script with torchrun..."
exec torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node="$SLURM_GPUS_ON_NODE" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --node_rank="$SLURM_NODEID" \
    "$@"
