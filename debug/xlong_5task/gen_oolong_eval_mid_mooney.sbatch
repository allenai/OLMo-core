#!/bin/bash
# oolong EVAL 8k/16k/32k regeneration on MOONEY (cubbins would not schedule: ReqNodeNotAvail behind
# larger pending jobs). One job, three rungs SEQUENTIALLY: the oolong-synth dataset downloads once
# into mooney's node-local HF cache, and a single process cannot trip the HF 429 rate-limit that
# killed two tasks when 28 array tasks hit the API at once.
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=mooney
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --job-name=xl5-oomid-mooney
#SBATCH --output=/data/prasann/xlong5_oomid_%A.log
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
source /usr/local/linux/miniforge-3.13/etc/profile.d/conda.sh 2>/dev/null || source /system/linux/miniforge-3.13/etc/profile.d/conda.sh
conda activate /data/prasann/conda/envs/corpus-reasoning-olmo
export PYTHONPATH=$REPO/src HF_HOME=/data/prasann/hf_cache HF_DATASETS_CACHE=/data/prasann/hf_cache/datasets
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
mkdir -p /data/prasann/xlong5/oolong_eval_gen
echo "=== HOST=$(hostname) START=$(date '+%F %T') ==="
set -- "8k 6963 8192 9801" "16k 13926 16384 9802" "32k 27852 32768 9803"
for spec in "$@"; do
  set -- $spec; LAB=$1; LO=$2; HI=$3; SEED=$4
  GEN=/data/prasann/xlong5/oolong_eval_gen/$LAB
  mkdir -p "$GEN"
  echo "--- rung=$LAB len=[$LO,$HI] seed=$SEED $(date '+%T') ---"
  python $REPO/src/corpus_reasoning/data/generate_oolong_ladder_data.py \
    --num-examples 2100 --len-min $LO --len-max $HI \
    --pool-max-ctx 262144 --tokenizer /scratch/users/prasann/hf_models/Qwen3.5-4B-Base \
    --seed $SEED --output-dir "$GEN"
  echo "  rc=$? $(date '+%T')  lines=$(cat $GEN/*.jsonl 2>/dev/null | wc -l)"
done
echo "=== DONE $(date '+%F %T') ==="
