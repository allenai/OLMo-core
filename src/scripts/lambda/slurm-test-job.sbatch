#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --output=/data/ai2/logs/%j.log

REPO_DIR=/data/ai2/petew/OLMo-core
VENV_DIR=/data/ai2/uv/OLMo-core
DATA_DIR=/data/caia-mltrain/data/

source "$VENV_DIR/bin/activate"
srun python -c "import torch; print(f'Torch version: {torch.__version__}'); print(f'Number of GPUs available: {torch.cuda.device_count()}')"
