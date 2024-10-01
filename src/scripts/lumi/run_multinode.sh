#!/usr/bin/env bash
#
# Script for running multinode jobs on LUMI. You can run this as a batch job using
# sbatch or as part of an interactive session by running this script as an executable.
#
# Example usage:
#
#      sbatch run_multinode.sh python src/train/OLMo-1B.py
#
#SBATCH --job-name=train-olmo
#SBATCH --account=project_462000229
#SBATCH --output=/pfs/lustref1/flash/project_462000229/logs/%j.log
#SBATCH --nodes=128             # Total number of nodes 
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8       # Allocate one gpu per MPI rank
#SBATCH --cpus-per-task=6
#SBATCH --time=48:00:00
#SBATCH --time-min=24:00:00
#SBATCH --mem=0			# All memory on the node
#SBATCH --partition=standard-g

module load LUMI/24.03 partition/G

## Container-dependent settings
export OLMO_CONTAINER=$PROJECT_DIR/containers/lumi-torch25rc-rocm62-py312.sif
export PYTHONPATH=.:${PYTHONPATH}
export ROCM_PATH=/opt/rocm
export CONDA_ENV=pytorch
# Setting LD_LIBRARY_PATH in the container allows us to leverage HPE cray libfabric on LUMI,
# which is optimized for high performancec computing. LUMI's containers already set up their
# path optimally.
#export SINGULARITYENV_LD_LIBRARY_PATH=/usr/local/lib:/opt/cray/libfabric/1.15.2.0/lib64

## General LUMI settings (these rarely change)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPICH_GPU_SUPPORT_ENABLED=1
export NCCL_SOCKET_IFNAME=hsn
export NCCL_NET_GDR_LEVEL=3
export MIOPEN_USER_DB_PATH=/tmp/${USER}-miopen-cache-${SLURM_JOB_ID}
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
export CXI_FORK_SAFE=1
export CXI_FORK_SAFE_HP=1
export FI_CXI_DISABLE_CQ_HUGETLB=1
export GPU_MAX_HW_QUEUES=8
# We need to set this to avoid "Cassini Event Queue overflow detected." errors.
export FI_CXI_DEFAULT_CQ_SIZE=131072
export OLMO_SHARED_FS=1

## Job settings
# Try playing with max_split_size_mb if you run into OOM errors.
#export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:128
# Disallow network calls for HF datasets. This can speed up loading datasets that are from the internet
export HF_DATASETS_OFFLINE=1
export DATA_PATH=$FLASH_DIR/preprocessed/olmo-mix
export CHECKPOINTS_PATH=$FLASH_DIR/checkpoints
export EVAL_DATA_PATH=$SCRATCH_DIR/eval-data

## Debug settings
#export NCCL_DEBUG=INFO
#export FI_LOG_LEVEL=INFO

srun \
  --cpus-per-task=$SLURM_CPUS_PER_TASK \
  --distribution=block:block \
  --kill-on-bad-exit \
  src/scripts/lumi/run_with_environment.sh \
    singularity exec \
    -B"$PROJECT_DIR:$PROJECT_DIR" \
    -B"$FLASH_DIR:$FLASH_DIR" \
    -B"$SCRATCH_DIR:$SCRATCH_DIR" \
    -B /var/spool/slurmd,/opt/cray/,/usr/lib64/libcxi.so.1,/usr/lib64/libjansson.so.4,/usr/lib64/libjson-c.so.3 \
    $OLMO_CONTAINER \
    src/scripts/lumi/run_in_container.sh ${@}
