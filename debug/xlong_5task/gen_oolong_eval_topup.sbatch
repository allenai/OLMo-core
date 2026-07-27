#!/bin/bash
# Top-up for the 2k/4k oolong eval rungs. Short examples have a small combinatorial space, so
# ~35-40% of a 600-example draw collides with the train pool or duplicates within the draw
# (measured: 2k kept 478/600, 4k kept 392/600). Generate ~4x more with fresh seeds; the finalizer
# dedups and filters against the train signature set.
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=cubbins
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=xl5-ooeval-top
#SBATCH --output=/data/prasann/xlong5/logs/ooeval_top_%a_%A.log
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
LAB=(2k 4k); LO=(1740 3480); HI=(2048 4096)
B=$(( SLURM_ARRAY_TASK_ID / 6 )); S=$(( SLURM_ARRAY_TASK_ID % 6 ))
GEN=/data/prasann/xlong5/oolong_eval_gen/${LAB[$B]}
mkdir -p "$GEN"
source /usr/local/linux/miniforge-3.13/etc/profile.d/conda.sh 2>/dev/null || source /system/linux/miniforge-3.13/etc/profile.d/conda.sh
conda activate /data/prasann/conda/envs/corpus-reasoning-olmo
export PYTHONPATH=$REPO/src HF_HOME=/data/prasann/hf_cache HF_DATASETS_CACHE=/data/prasann/hf_cache/datasets
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
echo "topup rung=${LAB[$B]} shard=$S"
python $REPO/src/corpus_reasoning/data/generate_oolong_ladder_data.py \
  --num-examples 400 --len-min ${LO[$B]} --len-max ${HI[$B]} \
  --pool-max-ctx 262144 --tokenizer /scratch/users/prasann/hf_models/Qwen3.5-4B-Base \
  --seed $(( 9500 + SLURM_ARRAY_TASK_ID )) --num-shards 6 --shard-index $S --output-dir "$GEN"
echo "rc=$? done $(date '+%F %T')"
