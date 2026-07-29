#!/bin/bash
# OOLONG ULTRA rungs 512k / 1M / 2M -- the oolong half of the >256k ladder extension.
#
# oolong is NOT a document pool, so build_xlong_rungs.py cannot expand it: it is a packed stream of
# labeled items, and a rung IS a token budget. So it regenerates through the normal synthetic
# generator at the new budget. Fully OFFLINE (synthetic item combinations from the cached
# oolong-synth dataset; HF_HUB_OFFLINE=1, no LLM, no BM25).
#
# ⚠ THE CAPACITY TRAP: the packer draws items WITHOUT REPLACEMENT from one dataset's item pool and
# stops when the budget is met -- if that pool holds fewer tokens than the target, the loop simply
# runs out of items and emits a SHORT example with no error. A 2M rung can therefore silently be a
# 700k rung. --pool-max-ctx is raised to admit the longest source records, and the realized lengths
# MUST be checked (check_bands.py) before any rung is published.
#
# Seeds 9500+ keep these item-combinations disjoint from the train pool (42-69) and from the
# earlier 2k/4k/64k/128k/256k eval build (9000-9019).
#
# Array = 3 rungs x 4 shards (id = rung*4 + shard), 150/shard => 600/rung, filtered to 300.
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=cubbins
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --job-name=xl5-ooultra
#SBATCH --output=/data/prasann/xlong5/logs/ooultra_%a_%A.log
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
LAB=(512k 1M 2M)
LO=(445645 891289 1782579)     # ~0.85 * target: the band floor, matching the 2k..256k build
HI=(524288 1048576 2097152)
B=$(( SLURM_ARRAY_TASK_ID / 4 )); S=$(( SLURM_ARRAY_TASK_ID % 4 ))
GEN=/data/prasann/xlong5/oolong_ultra_gen/${LAB[$B]}
mkdir -p "$GEN" /data/prasann/xlong5/logs

source /usr/local/linux/miniforge-3.13/etc/profile.d/conda.sh 2>/dev/null \
  || source /system/linux/miniforge-3.13/etc/profile.d/conda.sh
conda activate /data/prasann/conda/envs/corpus-reasoning-olmo
export PYTHONPATH=$REPO/src
export HF_HOME=/data/prasann/hf_cache HF_DATASETS_CACHE=/data/prasann/hf_cache/datasets
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1

echo "=== rung=${LAB[$B]} shard=$S len=[${LO[$B]},${HI[$B]}] HOST=$(hostname) $(date '+%F %T') ==="
python -u $REPO/src/corpus_reasoning/data/generate_oolong_ladder_data.py \
  --num-examples 150 --len-min ${LO[$B]} --len-max ${HI[$B]} \
  --pool-max-ctx 2097152 --tokenizer /scratch/users/prasann/hf_models/Qwen3.5-4B-Base \
  --seed $(( 9500 + SLURM_ARRAY_TASK_ID )) --num-shards 4 --shard-index $S --output-dir "$GEN"
echo "rc=$? done $(date '+%F %T')"
