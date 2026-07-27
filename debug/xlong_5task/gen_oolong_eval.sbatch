#!/bin/bash
# oolong EVAL rungs 2k/4k/64k/128k/256k -- these were never actually built (the old
# build_xlong_rungs.py only PRINTED the commands), and eval500_v2/oolong holds just 8k/16k/32k.
# Seeds are 9000+ so the synthetic item-combinations are disjoint from the train pool (seeds 42-69).
# Array = 5 rungs x 4 shards (id = rung*4 + shard), 150/shard => 600/rung, filtered to >=500.
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=cubbins
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --job-name=xl5-ooeval
#SBATCH --output=/data/prasann/xlong5/logs/ooeval_%a_%A.log
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
LAB=(2k 4k 64k 128k 256k)
LO=(1740 3480 55700 111400 222900)
HI=(2048 4096 65536 131072 262144)
B=$(( SLURM_ARRAY_TASK_ID / 4 )); S=$(( SLURM_ARRAY_TASK_ID % 4 ))
GEN=/data/prasann/xlong5/oolong_eval_gen/${LAB[$B]}
mkdir -p "$GEN" /data/prasann/xlong5/logs
source /usr/local/linux/miniforge-3.13/etc/profile.d/conda.sh 2>/dev/null || source /system/linux/miniforge-3.13/etc/profile.d/conda.sh
conda activate /data/prasann/conda/envs/corpus-reasoning-olmo
export PYTHONPATH=$REPO/src HF_HOME=/data/prasann/hf_cache HF_DATASETS_CACHE=/data/prasann/hf_cache/datasets
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
echo "rung=${LAB[$B]} shard=$S len=[${LO[$B]},${HI[$B]}]"
python $REPO/src/corpus_reasoning/data/generate_oolong_ladder_data.py \
  --num-examples 150 --len-min ${LO[$B]} --len-max ${HI[$B]} \
  --pool-max-ctx 262144 --tokenizer /scratch/users/prasann/hf_models/Qwen3.5-4B-Base \
  --seed $(( 9000 + SLURM_ARRAY_TASK_ID )) --num-shards 4 --shard-index $S --output-dir "$GEN"
echo "rc=$? done $(date '+%F %T')"
