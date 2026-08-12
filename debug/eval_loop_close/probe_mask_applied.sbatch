#!/bin/bash
#SBATCH --job-name=ctc-mask-probe
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=mooney
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0:30:00
#SBATCH --output=/data/prasann/joblogs/ctc-mask-probe-%j.log
set -euo pipefail
WORK=/data/prasann/ctc_eval_loop
export HOME=$WORK/home TMPDIR=$WORK/tmp TRITON_CACHE_DIR=$WORK/triton HF_HOME=$WORK/hf HF_HUB_OFFLINE=1
cd /accounts/projects/berkeleynlp/prasann/projects/newolmocore/OLMo-core
export PYTHONPATH=src:ctc/src
/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python debug/eval_loop_close/probe_mask_applied.py \
    --ckpt $WORK/ckpt/step30 --tokenizer $WORK/tok --data $WORK/data/mathmatch/eval_2k.jsonl
echo "=== PROBE EXIT=$? ==="
