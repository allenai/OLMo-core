#!/bin/bash
#SBATCH --job-name=ai2-slurm-test
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1

USERNAME=petew

# NOTE:
# The entrypoint script we use here will configure our training environment in the same way
# that our Beaker launcher does, and run the command you give it with 'torchrun'.
# It expects the following arguments:
# * Your (Beaker) username.
# * All other args are passed to torchrun.
srun ./src/scripts/lambda/entrypoint.sh $USERNAME ./src/scripts/lambda/distributed_test.py
