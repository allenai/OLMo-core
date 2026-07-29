#!/bin/bash
# ULTRA rungs 512k / 1M / 2M -- extends the xlong ladder past its 256k ceiling.
#
# Fully OFFLINE: build_xlong_rungs.py is doc-pool expansion over harvested fillers (no LLM, no
# BM25, no network). Pool capacity was verified in calibration before launching -- every task has
# >=2x headroom at 2M, so no example repeats a document (a within-example verbatim repeat is a
# known confound: see the free_pad_repeat / repeat_doc_text finding).
#
# Array = one task per index so the four build concurrently. Each index calibrates ONCE and then
# writes 512k, 1M, 2M.
#
# ⚠ Output goes to node-local /data, NOT /scratch. This build writes ~17 GB; /scratch is NFS at
# ~5 MB/s and would spend an hour in pure I/O. Sync to S3->weka afterwards (upload_to_s3.sbatch +
# weka_sync.sh), which is how the eval bundle reaches Beaker.
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=cubbins
#SBATCH --cpus-per-task=8
#SBATCH --mem=192G
#SBATCH --time=08:00:00
#SBATCH --job-name=xl5-ultra
#SBATCH --output=/data/prasann/xlong5/logs/ultra_%a_%A.log
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
TASKS=(contra nq outlier rerank)
# NOTE: --array=0 alone rebuilds ONLY contra (pubmed-only fillers, 2026-07-29).
T=${TASKS[$SLURM_ARRAY_TASK_ID]}
# OUT is overridable so a corrected rebuild lands in a CLEAN directory rather than sitting next to
# the files it supersedes: `n` changes whenever the filler pool changes, so a rebuilt rung gets a
# NEW filename and would otherwise coexist with the old one -- two files for one size, and
# sorted()[0] silently picks the rung.
OUT="${OUT:-/data/prasann/xlong5/eval_ultra}"
mkdir -p "$OUT" /data/prasann/xlong5/logs

source /usr/local/linux/miniforge-3.13/etc/profile.d/conda.sh 2>/dev/null \
  || source /system/linux/miniforge-3.13/etc/profile.d/conda.sh
conda activate /data/prasann/conda/envs/corpus-reasoning-olmo
export PYTHONPATH=$REPO/src:$REPO/src/scripts:$REPO/src/corpus_reasoning/data
export HF_HOME=/data/prasann/hf_cache HF_DATASETS_CACHE=/data/prasann/hf_cache/datasets
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1

echo "=== task=$T HOST=$(hostname) START=$(date '+%F %T') ==="
echo "--- /data free before build ---"; df -h /data | tail -1
# --count 500 clears the repo's eval_size>=500 floor. It is also the CEILING: the contra, nq and
# rerank canonicals hold exactly 500 rows, and gold-sanitization drops any example whose gold does
# not survive, so a task can legitimately land just under 500. Trust the "N ex" the builder prints,
# not this number, and report that realized count.
python -u $REPO/src/corpus_reasoning/data/build_xlong_rungs.py \
  --tasks "$T" --sizes 512k,1M,2M --count 500 \
  --tokenizer /scratch/users/prasann/hf_models/Qwen3.5-4B-Base \
  --out-root "$OUT"
rc=$?
echo "rc=$rc"
echo "--- built files ---"
ls -la "$OUT/$T"/ 2>/dev/null
echo "=== done $(date '+%F %T') ==="
exit $rc
