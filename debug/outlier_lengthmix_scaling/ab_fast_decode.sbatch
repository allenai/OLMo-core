#!/bin/bash
# GPU A/B validation of the new sparse-landmark fast decode: same checkpoint, 50 examples,
# OLMO_LANDMARK_SPARSE_DECODE=0 vs =1 (with the flag forced on via env), diff generations + time.
#SBATCH --job-name=ab-fastdecode
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:2
#SBATCH -w mooney
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=/data/prasann/joblogs/ab_fastdecode_%j.log
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
CR=/scratch/users/prasann/corpus-reasoning
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
export HOME=/data/prasann/home TMPDIR=/data/prasann/tmp TRITON_CACHE_DIR=/data/prasann/triton_cache
export EVAL500_ROOT=/data/prasann/outlier_lengthmix/eval500_local/outlier
cd "$CR"
CKPT=/data/prasann/ctc_suite/ckpts/lmx-slm-lr5e5-4b-loc
for MODE in 0 1; do
  OUT=/data/prasann/outlier_lengthmix/ab_decode_mode${MODE}.json
  echo "=== MODE=$MODE $(date '+%T') ==="
  OLMO_LANDMARK_SPARSE_DECODE=$MODE PYTHONPATH=$REPO/src:$CR \
  $PY -m torch.distributed.run --nproc_per_node=1 --master_port=29617 scripts/eval/eval_lc_native.py \
    --model-path "$CKPT" --out "$OUT" --tokenizer Qwen/Qwen3.5-0.8B --max-length 40960 \
    --max-test-samples 50 --batch-size 1 --skip-ruler --skip-gen \
    --landmark-mem-id 248200 --landmark-pad-id 248203 \
    --ladder --ladder-tasks outlier --ladder-rungs 3k \
    --outlier-data data/outlier_wiki100w_n55_k3_eval_600.jsonl || echo "MODE=$MODE FAILED"
done
echo "=== timing + diff ==="
$PY - <<'PYEOF'
import json
a = json.load(open("/data/prasann/outlier_lengthmix/ab_decode_mode0.json"))
b = json.load(open("/data/prasann/outlier_lengthmix/ab_decode_mode1.json"))
print("mode0 keys:", {k: v for k, v in a.items() if k in ("outlier_3k",)})
print("mode1 keys:", {k: v for k, v in b.items() if k in ("outlier_3k",)})
PYEOF
echo "=== DONE $(date '+%F %T') ==="
