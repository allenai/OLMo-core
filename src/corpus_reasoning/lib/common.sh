#!/bin/bash
# Shared environment setup and helper functions for training/eval shell scripts.
#
# Source this file in any script that needs CUDA/Triton setup or training launchers:
#   source "$PROJECT_DIR/scripts/lib/common.sh"
#   setup_env

setup_env() {
    # DeepSpeed and Flash Attention require CUDA_HOME to find nvcc.
    # This machine has CUDA 12.8 installed at /usr/local/cuda-12.8.
    export CUDA_HOME=/usr/local/cuda-12.8
    export PATH=$CUDA_HOME/bin:$PATH

    # Triton (used by Flash Attention) caches compiled kernels. Default location
    # is ~/.triton which can fill up home directory quotas — redirect to /tmp.
    export TRITON_CACHE_DIR=/tmp/triton_cache
    mkdir -p "$TRITON_CACHE_DIR"

    # huggingface_hub's default cache lives at $HF_HOME/hub (XDG-derived, per-node
    # /tmp on this cluster). Pin it to scratch so weights are NFS-shared across
    # the login node and every compute node — avoids re-downloading 7B-15B
    # checkpoints on each node and recovers from per-node /tmp pressure.
    export HF_HUB_CACHE=/scratch/users/prasann/huggingface-cache/hub
    mkdir -p "$HF_HUB_CACHE"
}

get_project_dir() {
    # Resolve the project root from the caller script's location.
    # Note: In sbatch scripts, BASH_SOURCE may not resolve correctly because
    # SLURM copies scripts to temp dirs — prefer setting PROJECT_DIR explicitly.
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
    dirname "$script_dir"
}

get_num_gpus() {
    # Use the NUM_GPUS env var if set, otherwise auto-detect from nvidia-smi.
    echo "${NUM_GPUS:-$(nvidia-smi -L | wc -l)}"
}

launch_training() {
    # Launch standard (non-chunked) training via Axolotl.
    # For chunked attention training, use train_chunked.py directly instead.
    local num_gpus=$1
    local config=$2
    accelerate launch --num_processes "$num_gpus" \
        -m axolotl.cli.train "$config"
}
