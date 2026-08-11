#!/bin/bash
#SBATCH --job-name=nq_ceconfirm_gen
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --account=site
#SBATCH --nodelist=mooney
#SBATCH --cpus-per-task=48
#SBATCH --mem=120G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/users/prasann/ctc_suite_logs/gen_nq_ceconfirm_%j.log

# CE-CONFIRM regen: EXACTLY the nq PILOT recipe (gen_nq_pilot_sharded.sbatch:
# k12-18, hard-neg-frac 0.1, seed 42) but WITH --ce-filter added. This isolates
# --ce-filter as the ONLY changed variable to test the mode-collapse diagnosis
# (pilot nq collapsed to [12]/[2], f1 0.074, because it was built WITHOUT
# --ce-filter -> false-negative hard negs -> gold not determinable).
# CE (MiniLM-L6) on CPU: k12-18 has <=1-2 hard negs/example so only ~few CE
# pairs/example -> fast on CPU, no GPU needed.
set -x
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export HF_HOME=/data/prasann/hf_cache
export HF_DATASETS_CACHE=/data/prasann/hf_cache
export HF_HUB_CACHE=/data/prasann/hf_cache/hub
export PYTHONWARNINGS=ignore
export PYTHONPATH=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/src
export CUDA_VISIBLE_DEVICES=""
PY=/scratch/users/prasann/conda/envs/corpus-reasoning-eval/bin/python
GEN=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/src/corpus_reasoning/data/generate_nq_training_data.py
OUT=/data/prasann/ctc_suite_data/nq_ceconfirm
LOGD=/data/prasann/ctc_suite_data/logs
mkdir -p "$OUT" "$LOGD"
cd /data/prasann || exit 1

NSHARDS=6
for i in $(seq 0 $((NSHARDS-1))); do
  $PY "$GEN" --num-train 440 --num-eval 0 --num-docs-min 12 --num-docs-max 18 \
    --hard-neg-frac 0.1 --ce-filter --pool-passages 20000 --pool-fetch-threads 8 --bm25-threads 4 \
    --num-shards $NSHARDS --shard-index $i \
    --output-dir "$OUT" --seed 42 \
    > "$LOGD/gen_nq_ceconfirm_s${i}of${NSHARDS}_${SLURM_JOB_ID}.log" 2>&1 &
done
wait
echo "DONE $(date)"
ls -la "$OUT"
echo "--- line counts ---"
wc -l "$OUT"/*.jsonl 2>/dev/null