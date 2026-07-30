#!/bin/bash
# oolong 2k->256k banded. Array = 7 bands x 4 shards (id = band*4 + shard).
# oolong is token-budgeted directly (--len-min/--len-max ARE the context budget), so there is no
# doc-count calibration; --pool-max-ctx is raised 131072 -> 262144 so long source windows can
# contribute items. NO --item-regex anywhere downstream (see the leak fix).
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=cubbins
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --job-name=xl5-oolong
#SBATCH --output=/data/prasann/xlong5/logs/oolong_%a_%A.log
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
LO=(2048 4096 8192 16384 32768 65536 131072)
HI=(3973 7946 15892 31784 63569 127139 254280)
PER=(1500 1250 1000 675 350 150 75)   # per-shard counts (4 shards/band)
B=$(( SLURM_ARRAY_TASK_ID / 4 )); S=$(( SLURM_ARRAY_TASK_ID % 4 ))
OUT=/data/prasann/xlong5/pools/oolong/band${B}_shard${S}
mkdir -p "$OUT" /data/prasann/xlong5/logs
source /usr/local/linux/miniforge-3.13/etc/profile.d/conda.sh 2>/dev/null || source /system/linux/miniforge-3.13/etc/profile.d/conda.sh
conda activate /data/prasann/conda/envs/corpus-reasoning-olmo
export PYTHONPATH=$REPO/src HF_HOME=/data/prasann/hf_cache HF_DATASETS_CACHE=/data/prasann/hf_cache/datasets
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
# 28 concurrent tasks hitting the HF dataset API tripped a 429 IP rate-limit (tasks 5,23 died).
# The dataset is cached under HF_HOME after the first success, so force offline reuse.
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
echo "band=$B shard=$S len=[${LO[$B]},${HI[$B]}] n=${PER[$B]}"
python $REPO/src/corpus_reasoning/data/generate_oolong_ladder_data.py \
  --num-examples ${PER[$B]} --len-min ${LO[$B]} --len-max ${HI[$B]} \
  --pool-max-ctx 262144 --tokenizer /scratch/users/prasann/hf_models/Qwen3.5-4B-Base \
  --seed $(( 42 + SLURM_ARRAY_TASK_ID )) --num-shards 4 --shard-index $S --output-dir "$OUT"
echo "rc=$? done $(date '+%F %T')"
