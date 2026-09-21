#!/bin/bash
# Cost model for the CTC suite, 2k-32k first (longer rungs come after -- they need a YaRN
# serving copy at 256k and the measurement here sizes what is worth running up there).
# 7 tasks spanning the decode-budget spectrum (retrieval 64 -> grouping 4096) x 5 rungs x 100 ex.
#SBATCH --job-name=ctcfast-bench
#SBATCH --account=site
#SBATCH --partition=berkeleynlp
#SBATCH --qos=preemptive
#SBATCH --nodelist=horton
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/data/prasann/joblogs/ctcfast_bench_%j.log
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
PY=/data/prasann/ctc_vllm_venv/bin/python
SERVE=/data/prasann/ctc_suite/vllm_serving_4b_v3/ctc-s5-contra-full-4b
DATA=/net/cubbins/data/prasann/ctc_suite_staged/eval_rungs
OUT=/data/prasann/ctc_fast_suite/bench_2k32k_${SLURM_JOB_ID}.json

# Node-local caches: HOME on NFS wedges the flashinfer/triton JIT, and nvcc leaks ~165MB/job
# into TMPDIR (see records: flashinfer-cache-home-nfs-wedge, gdn-jit-tmp-disk-fill).
export HOME=/data/prasann/jithome
export FLASHINFER_CACHE_DIR=/data/prasann/flashinfer_cache
export TMPDIR=/data/prasann/tmp
export TRITON_CACHE_DIR=/data/prasann/triton_cache
mkdir -p "$HOME" "$FLASHINFER_CACHE_DIR" "$TMPDIR" "$TRITON_CACHE_DIR" /data/prasann/ctc_fast_suite
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1

# subset:spec:rung_tokens -- contradiction_iid's 2k rung is the 2560 file (its 2048 build sat
# below the training minimum; the alias is explicit in the olmo-eval roster too).
CELLS=""
for t in 2048 4096 8192 16384 32768; do
  for pair in fiqa:retrieval outlier:outlier oolong:oolong qdmatch_nq:qdmatch textgroups:textgroups grouping:grouping; do
    CELLS="$CELLS,${pair%%:*}:${pair##*:}:$t"
  done
  c=$t; [ "$t" = 2048 ] && c=2560
  CELLS="$CELLS,contradiction_iid:contradiction:$c"
done
CELLS="${CELLS#,}"

echo "=== HOST=$(hostname) START=$(date '+%F %T') ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY -u "$REPO/debug/ctc_fast_suite/bench_vllm_rungs.py" \
  --model "$SERVE" --data-root "$DATA" --cells "$CELLS" \
  --limit 100 --max-model-len 40960 --out "$OUT"
rc=$?
mkdir -p "$REPO/debug/ctc_fast_suite/results"
cp "$OUT" "$REPO/debug/ctc_fast_suite/results/bench_2k32k.json" 2>/dev/null
echo "=== DONE rc=$rc $(date '+%F %T') ==="
exit $rc
