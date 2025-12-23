#!/bin/bash
#SBATCH --job-name=ai2-slurm-test
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --output=/data/ai2/logs/%j.log

REPO_DIR=/data/ai2/petew/OLMo-core
VENV_DIR=/data/ai2/uv/OLMo-core-petew
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
echo "Using $(python --version) at $(which python) with the following packages:"
uv pip freeze

echo "============= Setup complete ============="

python -c "import torch; print(f'Torch version: {torch.__version__}'); print(f'Number of GPUs available: {torch.cuda.device_count()}')"
