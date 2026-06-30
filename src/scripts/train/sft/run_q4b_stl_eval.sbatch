#!/bin/bash
# Reusable NATIVE eval for LOCAL Qwen3-4B **dense plain-attention** single-task-ladder SFT
# checkpoints (the *-stl_validation / Qwen3-4B-singletask-ladder-SFT-local.py runs). Plain dense
# (flash-attn2 + YaRN) can use the standard olmo_core generate path (gm.generate_batch) -- NO
# document-chunked attention -- so we eval with scripts/eval/eval_lc_native.py loaded directly from
# the local distcp checkpoint (no HF export, no weka, no Beaker). One jsteinhardt H200 node; runs
# data-parallel across 8 GPUs (each rank a full model copy + a shard of the examples).
#
# Inputs via --export (TASK + CKPT required; MAX_TEST optional):
#   sbatch --job-name=stl-eval-contra \
#     --export=ALL,TASK=contra,CKPT=/data/prasann/singletask_ladder_runs/q4b-dense-contra-14336val/step400 \
#     src/scripts/train/sft/run_q4b_stl_eval.sbatch
#
#   TASK = contra | nq | rerank | outlier | oolong
#   CKPT = a STEP DIR (has config.json + model_and_optim/.metadata)
#   MAX_TEST = #eval examples (default 100)
#
# Writes: <run-dir>/eval/<task>_14k.json   (run-dir = parent of the step dir)
#
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive
#SBATCH -w mooney
#SBATCH --gres=gpu:H200:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=02:00:00
#SBATCH --output=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/.stleval_%x_%j.log

set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
CR=/scratch/users/prasann/corpus-reasoning
NGPU="${NGPU:-8}"
TASK="${TASK:?set TASK=contra|nq|rerank|outlier|oolong}"
CKPT="${CKPT:?set CKPT=<step dir with config.json + model_and_optim/>}"
MAX_TEST="${MAX_TEST:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_LENGTH="${MAX_LENGTH:-14336}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen3-4B}"

# Strip a trailing slash, derive run dir + output path.
CKPT="${CKPT%/}"
RUN_DIR="$(dirname "$CKPT")"
RUN="$(basename "$RUN_DIR")"
OUT="$RUN_DIR/eval/${TASK}_14k.json"
echo "=== HOST=$(hostname) RUN=$RUN TASK=$TASK CKPT=$CKPT MAX_TEST=$MAX_TEST MAXLEN=$MAX_LENGTH START=$(date '+%F %T') ==="

if [ ! -f "$CKPT/model_and_optim/.metadata" ]; then
  echo "ERROR: $CKPT/model_and_optim/.metadata not found -- not a complete step dir"; exit 2
fi

source /system/linux/miniforge-3.13/etc/profile.d/conda.sh
OLMOENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$OLMOENV" ] || OLMOENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo

export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=lo

# Trust SLURM's GPU allocation (nvidia-smi is not cgroup-isolated here): pick clean GPUs.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  mapfile -t CLEAN < <(nvidia-smi --query-gpu=memory.used,uuid --format=csv,noheader,nounits | awk -F', ' '$1+0 < 2000 {print $2}')
  export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${CLEAN[*]:0:$NGPU}")
fi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

conda activate "$OLMOENV"
# olmo-core src (custom modules) shadows the editable install; corpus-reasoning for scripts.eval.*
export PYTHONPATH="$REPO/src:$CR"
cd "$CR"   # relative data/... paths resolve against the corpus-reasoning checkout

COMMON="--model-path $CKPT --out $OUT --tokenizer $TOKENIZER --max-length $MAX_LENGTH \
        --max-test-samples $MAX_TEST --batch-size $BATCH_SIZE"

# Per-task eval. eval_lc_native.py auto-detects the (plain dense) variant from config.json.
#  * contra -> non-ladder isolated path: full result dict (f1 + parse_rate). ruler/nq/gen skipped;
#    nq pointed at a nonexistent file so only contradiction runs. short-cot/template budget = 512.
#  * nq/rerank/outlier/oolong -> single-rung --ladder (rerank+outlier exist ONLY in the ladder map;
#    nq isolated there too). Each rung's data file is the one chosen to sit <=14k tokens.
case "$TASK" in
  contra)
    ARGS="$COMMON --skip-ruler --skip-gen --nq-data /nonexistent_skip_nq \
          --contra-data data/contradiction_eval_pubmed_both_n100_k3.jsonl \
          --contra-max-new-tokens 512" ;;
  nq)
    ARGS="$COMMON --skip-ruler --skip-gen --ladder --ladder-tasks nq --ladder-rungs 3k \
          --nq-data data/n2ified_eval_nq_q50.jsonl" ;;
  rerank)
    ARGS="$COMMON --skip-ruler --skip-gen --ladder --ladder-tasks rerank --ladder-rungs 2k \
          --rerank-data data/msmarco_dev_rerank_k50_1000.jsonl" ;;
  outlier)
    ARGS="$COMMON --skip-ruler --skip-gen --ladder --ladder-tasks outlier --ladder-rungs 3k \
          --outlier-data data/outlier_wiki100w_n55_k3_eval_100.jsonl" ;;
  oolong)
    # plain-dense oolong: ladder oolong uses the standard generate path on the raw prompt (NO box
    # markers / chunked mask), unlike eval_lc_native_docchunk.py which is for document-chunked models.
    ARGS="$COMMON --skip-ruler --skip-gen --ladder --ladder-tasks oolong --ladder-rungs 8k" ;;
  *)
    echo "ERROR: unknown TASK=$TASK (want contra|nq|rerank|outlier|oolong)"; exit 2 ;;
esac

echo "=== EVAL $(date '+%T') -> $OUT ==="
EVAL_PORT=$(( 20000 + RANDOM % 20000 ))
torchrun --nproc_per_node="$NGPU" --master_port="$EVAL_PORT" \
  scripts/eval/eval_lc_native.py $ARGS
rc=$?
echo "eval rc=$rc result=$OUT"
[ -f "$OUT" ] && { echo "--- $OUT ---"; cat "$OUT"; }
echo "=== DONE rc=$rc $(date '+%F %T') ==="
exit $rc
