#!/bin/bash
#SBATCH --job-name=vllm-canon-fast
#SBATCH --partition=berkeleynlp
#SBATCH --qos=preemptive
#SBATCH --nodelist=horton
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/data/prasann/joblogs/smoke_canon_fast_%j.log
set -x
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
REPO_ROOT=${REPO_ROOT:-$(git -C "${SLURM_SUBMIT_DIR:-$(dirname "$0")}" rev-parse --show-toplevel)}
/data/prasann/ctc_vllm_venv/bin/python $REPO_ROOT/research/landmark/ctc_vllm_validation/smoke_canonical_fast.py \
  /data/prasann/ctc_suite/hf_exports_4b/ctc-s5-contra-full-4b
