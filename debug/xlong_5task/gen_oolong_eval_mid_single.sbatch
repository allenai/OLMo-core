#!/bin/bash
# oolong EVAL 8k/16k/32k regeneration, ONE job per rung (not an array).
# An 18-task array could not backfill on cubbins behind larger pending jobs; single modest jobs
# schedule immediately. Each invocation loads the oolong-synth cache once and emits the whole rung.
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=cubbins
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --job-name=xl5-oomid
#SBATCH --output=/data/prasann/xlong5/logs/oomid_%x_%A.log
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
: "${LAB:?}"; : "${LO:?}"; : "${HI:?}"; : "${SEED:?}"
GEN=/data/prasann/xlong5/oolong_eval_gen/$LAB
mkdir -p "$GEN"
source /usr/local/linux/miniforge-3.13/etc/profile.d/conda.sh 2>/dev/null || source /system/linux/miniforge-3.13/etc/profile.d/conda.sh
conda activate /data/prasann/conda/envs/corpus-reasoning-olmo
export PYTHONPATH=$REPO/src HF_HOME=/data/prasann/hf_cache HF_DATASETS_CACHE=/data/prasann/hf_cache/datasets
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
echo "rung=$LAB len=[$LO,$HI] seed=$SEED start=$(date '+%T')"
python $REPO/src/corpus_reasoning/data/generate_oolong_ladder_data.py \
  --num-examples 2100 --len-min $LO --len-max $HI \
  --pool-max-ctx 262144 --tokenizer /scratch/users/prasann/hf_models/Qwen3.5-4B-Base \
  --seed $SEED --output-dir "$GEN"
echo "rc=$? done $(date '+%F %T')"
