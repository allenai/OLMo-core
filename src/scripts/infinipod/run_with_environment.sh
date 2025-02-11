#!/usr/bin/env bash
set -exuo pipefail

CONDA_ENV=$1
shift

# Redirect stdout and stderr so that we get a prefix with the node name
export NODENAME=$(hostname -s)
exec > >(trap "" INT TERM; sed -u "s/^/$NODENAME:$SLURM_LOCALID out: /")
exec 2> >(trap "" INT TERM; sed -u "s/^/$NODENAME:$SLURM_LOCALID err: /" >&2)

# Read secrets from env file
set -a
set +x
source /home/common/shanea/.env
set -x
set +a

# Move AWS credentials from env to relevant files
mkdir -p ~/.aws
printenv AWS_CONFIG > ~/.aws/config
printenv AWS_CREDENTIALS > ~/.aws/credentials

# Set up conda
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV
pip freeze

# Infinipod specific environment
export NCCL_FASTRAK_LLCM_DEVICE_DIRECTORY="/dev/aperture_devices"
NCCL_LIB_DIR="/usr/local/nvidia/lib64" source /usr/local/nvidia/lib64/nccl-env-profile.sh
export LD_LIBRARY_PATH=/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu:"$LD_LIBRARY_PATH"
export NCCL_SHIMNET_SHIM_LAYERS="unused"
export NCCL_TUNER_PLUGIN="none"
export NVIDIA_PYTORCH_VERSION=24.03
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Setup for multi-node
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=34126 # This can be any free port

# Force processes to synchronize at init_process_group
export TORCH_DIST_INIT_BARRIER=1
# Better error handling from Python
export PYTHONFAULTHANDLER=1

# Set the temp dir
export TMPDIR=/mnt/localdisk/tmp
mkdir -p $TMPDIR

exec ${@}
