#!/bin/bash
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=cubbins
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --job-name=xl5-exp
#SBATCH --output=/data/prasann/xlong5/logs/%x_%a_%A.log
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
: "${TASK:?set TASK=nq|outlier|rerank}"
mkdir -p /data/prasann/xlong5/logs /data/prasann/xlong5/pools/$TASK
source /usr/local/linux/miniforge-3.13/etc/profile.d/conda.sh 2>/dev/null || source /system/linux/miniforge-3.13/etc/profile.d/conda.sh
conda activate /data/prasann/conda/envs/corpus-reasoning-olmo
export PYTHONPATH=$REPO/src HF_HOME=/data/prasann/hf_cache TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
python $REPO/debug/xlong_5task/gen_banded_expand.py \
  --task "$TASK" \
  --out /data/prasann/xlong5/pools/$TASK/${TASK}_banded_2k-256k_shard${SLURM_ARRAY_TASK_ID}.jsonl \
  --num-shards 8 --shard-index ${SLURM_ARRAY_TASK_ID} --pool-src 3000
echo "rc=$? done $(date '+%F %T')"
