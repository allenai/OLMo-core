#!/bin/bash
# Re-eval the trained landmark 5-task checkpoint at a chosen landmark top-k fraction, to test
# whether the depressed RULER (0.749 at top_k=0.1) is a sparsity artifact. RULER + contra + nq
# (skip-extra) at n=50, the same 2 contexts as the baseline run. Must run on mooney (ckpt is on
# mooney node-local /data). 1 GPU.
#   sbatch --job-name=lmtk-10 --nodelist=mooney --qos=preemptive_high --gres=gpu:H200:1 \
#     --export=ALL,TOPK=1.0 src/scripts/train/sft/eval_landmark_topk.sbatch
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=mooney
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=03:00:00
#SBATCH --output=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/.lmtk_%x_%j.log
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
CR=/scratch/users/prasann/corpus-reasoning
TOPK="${TOPK:?set TOPK}"
CKPT=/data/prasann/cpt_mix_runs/q4b-lm-5wF-c85/step1465
source /system/linux/miniforge-3.13/etc/profile.d/conda.sh
OLMOENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$OLMOENV" ] || OLMOENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
conda activate "$OLMOENV"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/src:$CR"
cd "$CR"
echo "HOST=$(hostname) CUDA=$CUDA_VISIBLE_DEVICES TOPK=$TOPK START=$(date '+%T')"
python -u scripts/eval/eval_lc_native_landmark.py \
  --model-path "$CKPT" \
  --out "outputs/eval_results/q4b-lm-5wF-c85_topk${TOPK}.json" \
  --tokenizer Qwen/Qwen3-4B --max-length 16384 \
  --landmark-top-k-fraction "$TOPK" \
  --ruler-lengths L1024,L2048 --max-test-samples 50 --skip-extra
echo "DONE $(date '+%T') -> outputs/eval_results/q4b-lm-5wF-c85_topk${TOPK}.json"
