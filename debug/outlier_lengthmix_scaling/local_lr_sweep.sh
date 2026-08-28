#!/bin/bash
# LOCAL LR sweep on mooney: repair base node-locally, then submit all 6 LR runs as 4-GPU
# jsteinhardt jobs (the 8-GPU QOS cap runs them 2 at a time).
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
cd "$REPO"

echo "[local] step 1: repair base on mooney (node-local read+write)"
srun --account=site --partition=jsteinhardt --qos=preemptive_high --nodelist=mooney \
  --job-name=q35fix-local --cpus-per-task=16 --mem=150G --time=01:00:00 bash -c '
export HOME=/data/prasann/home HF_HOME=/data/prasann/hf_cache TMPDIR=/data/prasann/tmp
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
[ -x "$PY" ] || PY=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python
[ -s /data/prasann/ctc_suite/bases/q35-4b-base-markerfix/model_and_optim/.metadata ] && { echo "[skip] markerfix exists"; exit 0; }
PYTHONPATH='"$REPO"'/src $PY '"$REPO"'/src/scripts/data/fix_marker_embeddings_qwen35.py \
  --base /data/prasann/ctc_suite/bases/q35-4b-base-modelonly/model_and_optim \
  --out  /data/prasann/ctc_suite/bases/q35-4b-base-markerfix \
  --model-scale 4b
' || { echo "[local] repair FAILED"; exit 1; }

echo "[local] step 2: submit 6 LR runs (4 GPUs each)"
cd "$REPO/src/scripts/train/memexpress/ctc_suite"
for variant in full sparselandmark; do
  vtag=full; [ "$variant" = "sparselandmark" ] && vtag=slm
  for lr in 2e-5 5e-5 1.2e-4; do
    case "$lr" in 2e-5) tag=lr2e5;; 5e-5) tag=lr5e5;; 1.2e-4) tag=lr1p2e4;; esac
    RUN="lmx-${vtag}-${tag}-4b-loc"
    PARTITION=jsteinhardt QOS=preemptive_high ACCOUNT=site NODE=mooney TIME=03:00:00 \
    TASK=outlier DATA_SRC=/data/prasann/outlier_lengthmix/arms_tokenized/lr2k5000 \
    VARIANT="$variant" SCALE=4b MODEL_FAMILY=qwen3_5 RUN="$RUN" EPOCHS=1 SEQ_LEN=4096 LR="$lr" \
    GLOBAL_BATCH=8 MICRO_BATCH=1 NGPU=8 \
    BASE_SRC=/data/prasann/ctc_suite/bases/q35-4b-base-markerfix \
    WANDB_GROUP=outlier-lengthmix-lr \
    ./run_ctc_local.sbatch
    sleep 2
  done
done
echo "[local] all 6 submitted"
squeue -u prasann -h -o "%8i %18j %4t %R" | grep -i lmx || true
