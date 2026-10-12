#!/bin/bash
#SBATCH --job-name=vllm-smoke-v3c
#SBATCH --account=site
#SBATCH --partition=berkeleynlp
#SBATCH --qos=preemptive
#SBATCH --nodelist=horton
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH --output=/data/prasann/joblogs/vllm_smoke_v3c_%j.log
set -uo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# FlashInfer JIT-compiles the Qwen GDN prefill kernel and needs a real CUDA toolkit;
# vLLM's default probe (/usr/local/cuda) is absent on horton, but 12.8 (matches torch cu128) is.
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
PY=/data/prasann/ctc_vllm_venv/bin/python
REPO_ROOT=${REPO_ROOT:-$(git -C "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" rev-parse --show-toplevel)}
VDIR=$REPO_ROOT/research/landmark/ctc_vllm_validation

echo "=== HOST=$(hostname) START=$(date '+%F %T') nvcc=$(which nvcc) ==="
$PY -u "$VDIR/smoke_load_v3.py"
echo "=== DONE $(date '+%F %T') ==="
