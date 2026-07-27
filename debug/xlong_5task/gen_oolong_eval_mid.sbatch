#!/bin/bash
# oolong EVAL rungs 8k/16k/32k, REGENERATED disjoint from the new 2k->256k train pool.
# The eval500_v2 originals were filtered only against the *v1* oolong train split, and measured
# 21.6% / 11.4% / 3.6% overlap with the new pool -- enough to inflate oolong at those rungs.
# Seeds 9800+ (train used 42-69, other eval rungs 9000-9511). Over-generate ~4x: short rungs lose
# 30-40% to train-overlap + intra-draw duplicates.
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=cubbins
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:25:00
#SBATCH --job-name=xl5-ooeval-mid
#SBATCH --output=/data/prasann/xlong5/logs/ooeval_mid_%a_%A.log
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
LAB=(8k 16k 32k); LO=(6963 13926 27852); HI=(8192 16384 32768)
B=$(( SLURM_ARRAY_TASK_ID / 6 )); S=$(( SLURM_ARRAY_TASK_ID % 6 ))
GEN=/data/prasann/xlong5/oolong_eval_gen/${LAB[$B]}
mkdir -p "$GEN"
source /usr/local/linux/miniforge-3.13/etc/profile.d/conda.sh 2>/dev/null || source /system/linux/miniforge-3.13/etc/profile.d/conda.sh
conda activate /data/prasann/conda/envs/corpus-reasoning-olmo
export PYTHONPATH=$REPO/src HF_HOME=/data/prasann/hf_cache HF_DATASETS_CACHE=/data/prasann/hf_cache/datasets
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
echo "rung=${LAB[$B]} shard=$S len=[${LO[$B]},${HI[$B]}]"
python $REPO/src/corpus_reasoning/data/generate_oolong_ladder_data.py \
  --num-examples 350 --len-min ${LO[$B]} --len-max ${HI[$B]} \
  --pool-max-ctx 262144 --tokenizer /scratch/users/prasann/hf_models/Qwen3.5-4B-Base \
  --seed $(( 9800 + SLURM_ARRAY_TASK_ID )) --num-shards 6 --shard-index $S --output-dir "$GEN"
echo "rc=$? done $(date '+%F %T')"
