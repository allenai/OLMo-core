#!/bin/bash
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=cubbins
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --job-name=xl5-contra
#SBATCH --output=/data/prasann/xlong5/logs/contra_%a_%A.log
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
mkdir -p /data/prasann/xlong5/logs /data/prasann/xlong5/pools/contradiction
source /usr/local/linux/miniforge-3.13/etc/profile.d/conda.sh 2>/dev/null || source /system/linux/miniforge-3.13/etc/profile.d/conda.sh
conda activate /data/prasann/conda/envs/corpus-reasoning-olmo
export PYTHONPATH=$REPO/src HF_HOME=/data/prasann/hf_cache HF_DATASETS_CACHE=/data/prasann/hf_cache/datasets
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
python $REPO/debug/xlong_5task/gen_banded_contra.py \
  --src /data/prasann/ctc_suite_data/contradiction_pool/contradiction_train_pubmed_realistic_n50-950_k3.jsonl \
  --out /data/prasann/xlong5/pools/contradiction/contra_banded_2k-256k_shard${SLURM_ARRAY_TASK_ID}.jsonl \
  --pool-abstracts 80000 --num-shards 8 --shard-index ${SLURM_ARRAY_TASK_ID} --limit-src 4000
echo "rc=$? done $(date '+%F %T')"
