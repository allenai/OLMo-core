#!/bin/bash
# Rebuild the 64k/128k/256k eval rungs at eval_size=500 (the previous build used --count 300,
# below the >=500 minimum -- see eval500_v2/_xlong_build.log). Qwen3.5 tokenizer for calibration.
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=cubbins
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --job-name=xl5-eval
#SBATCH --output=/data/prasann/xlong5/logs/eval_xlong_%A.log
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
OUT=/data/prasann/xlong5/eval
mkdir -p "$OUT"/{contra,nq,outlier,rerank,oolong} /data/prasann/xlong5/logs
source /usr/local/linux/miniforge-3.13/etc/profile.d/conda.sh 2>/dev/null || source /system/linux/miniforge-3.13/etc/profile.d/conda.sh
conda activate /data/prasann/conda/envs/corpus-reasoning-olmo
# ctc_eval (used by build_xlong_rungs._measure_ctx) lives under src/scripts, not src.
export PYTHONPATH=$REPO/src:$REPO/src/scripts:$REPO/src/corpus_reasoning/data HF_HOME=/data/prasann/hf_cache
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
python $REPO/src/corpus_reasoning/data/build_xlong_rungs.py \
  --tasks contra,nq,outlier,rerank --sizes 64k,128k,256k --count 500 \
  --out-root "$OUT" --tokenizer /scratch/users/prasann/hf_models/Qwen3.5-4B-Base
echo "rc=$? done $(date '+%F %T')"
find "$OUT" -name "*.jsonl" | while read f; do echo "$(wc -l < "$f") $f"; done
