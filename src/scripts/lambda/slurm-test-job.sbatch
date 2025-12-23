#!/bin/bash
#SBATCH --job-name=ai2-slurm-test
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --output=/data/ai2/logs/%j/task_%t.log

USERNAME=petew
WORKSPACE=ai2/linear-rnns

# NOTE:
# * Beaker username must be the first argument to the entrypoint script.
# * Beaker workspace the second argument.
# * All other args are executed as-is.
./src/scripts/lambda/entrypoint.sh $USERNAME $WORKSPACE python -c "$(cat <<'EOF'
import torch
print(f"Torch version: {torch.__version__}")
print(f"Number of GPUs available: {torch.cuda.device_count()}")
print("Done!")
EOF
)"
