#!/bin/bash
# Local FAST dense ladder eval via vLLM (corpus-reasoning-eval env has vllm 0.18.1; local flash works).
# Evals the two exported dense HF checkpoints (32k + 64k) across the length ladder to 64k.
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#SBATCH --output=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/.densevllm_%j.log
set -uo pipefail
CR=/scratch/users/prasann/corpus-reasoning
HF=/scratch/users/prasann/dense_eval_hf
source /system/linux/miniforge-3.13/etc/profile.d/conda.sh
conda activate /scratch/users/prasann/conda/envs/corpus-reasoning-eval
export PYTHONPATH="$CR" TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
cd "$CR"
RULER="${RULER_LENGTHS:-L1024,L2048,L4096,L8192,L16384}"
N="${NSAMP:-50}"
for RUN in q4b-cptmix-5task-32k-am q4b-cptmix-5task-64k-am; do
  echo "===== LOCAL vLLM ladder eval $RUN $(date '+%T') ====="
  python -u scripts/eval/eval_lc_fast.py \
    --base-model "$HF/${RUN}_hf" \
    --ladder --max-model-len 65536 --max-test-samples "$N" \
    --ruler-lengths "$RULER" \
    --out "$CR/outputs/eval_results/${RUN}_ladder_local.json"
  echo "rc=$? $RUN done $(date '+%T')"
done
echo "===== ALL DENSE LOCAL EVALS DONE $(date '+%T') ====="
