#!/bin/bash
#SBATCH --job-name=ai2-slurm-test
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --output=/data/ai2/logs/%j/task_%t.log

USERNAME=petew
WORKSPACE=ai2/linear-rnns

# NOTE:
# The entrypoint script we use here will configure our training environment in the same way
# that our Beaker launcher does, and run the command you give it with 'torchrun'.
# It expects the following arguments:
# * Your Beaker username.
# * A Beaker workspace to pull env variables from.
# * All other args are passed to torchrun.
./src/scripts/lambda/entrypoint.sh $USERNAME $WORKSPACE python -c "$(cat <<'EOF'
import torch
print(f"Torch version: {torch.__version__}")
print(f"Number of GPUs available: {torch.cuda.device_count()}")
print("Done!")
EOF
)"
