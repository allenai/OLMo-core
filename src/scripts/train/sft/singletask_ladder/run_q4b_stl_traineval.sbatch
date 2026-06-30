#!/bin/bash
# NODE-PORTABLE single-task-ladder VALIDATION: train (dense, 14k-capped) THEN eval, self-contained on
# whatever H200 node SLURM places it. Data is read from SHARED /scratch (so any node works); the
# checkpoint is written to that node's LOCAL /data; eval runs on the same node; the result JSON is
# copied back to SHARED /scratch for central collection. Submit across nodes/partitions to parallelize.
#
#   sbatch -p jsteinhardt --job-name=te-oolong --export=ALL,TASK=oolong src/scripts/train/sft/singletask_ladder/run_q4b_stl_traineval.sbatch
#   sbatch -p berkeleynlp --job-name=te-outlier --export=ALL,TASK=outlier src/scripts/train/sft/singletask_ladder/run_q4b_stl_traineval.sbatch
#
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive
#SBATCH --gres=gpu:H200:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=04:00:00
#SBATCH --output=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/.stlte_%x_%j.log
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
CR=/scratch/users/prasann/corpus-reasoning
NGPU="${NGPU:-8}"
TASK="${TASK:?set TASK=contra|nq|rerank|outlier|oolong}"
SEQ_LEN="${SEQ_LEN:-14336}"
BATCH_TOKENS="${BATCH_TOKENS:-57344}"
MAX_STEPS="${MAX_STEPS:-500}"
SAVE_INTERVAL="${SAVE_INTERVAL:-200}"
LR="${LR:-1e-5}"
MAX_TEST="${MAX_TEST:-100}"
OUTLIER_EVAL_DATA="${OUTLIER_EVAL_DATA:-data/outlier_wiki100w_n55_k3_eval_100.jsonl}"  # override per seq_len (n20≈3k, n55≈8.6k)
DATA_ROOT="${DATA_ROOT:-/scratch/users/prasann/stl_14k}"     # SHARED
RUN_NAME="${RUN_NAME:-q4b-dense-${TASK}-14kcap}"
SAVE_FOLDER=/data/prasann/singletask_ladder_runs/$RUN_NAME    # NODE-LOCAL
WBGROUP="${WBGROUP:-q4b-stl-14kcap}"
RESULTS_SHARED=/scratch/users/prasann/stl_eval_results        # SHARED collection point
echo "=== HOST=$(hostname) RUN=$RUN_NAME TASK=$TASK SEQ=$SEQ_LEN BT=$BATCH_TOKENS STEPS=$MAX_STEPS START=$(date '+%F %T') ==="

source /system/linux/miniforge-3.13/etc/profile.d/conda.sh
OLMOENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$OLMOENV" ] || OLMOENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=lo
export WANDB_API_KEY=$(awk '/api.wandb.ai/{getline; if($1=="login")getline; for(i=1;i<=NF;i++) if($i=="password") print $(i+1)}' ~/.netrc 2>/dev/null | head -1)
[ -z "${WANDB_API_KEY:-}" ] && export WANDB_API_KEY=$(grep -A2 'machine api.wandb.ai' ~/.netrc | awk '/password/{print $2}' | head -1)
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  mapfile -t CLEAN < <(nvidia-smi --query-gpu=memory.used,uuid --format=csv,noheader,nounits | awk -F', ' '$1+0 < 2000 {print $2}')
  export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${CLEAN[*]:0:$NGPU}")
fi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
conda activate "$OLMOENV"

# ---------- TRAIN ----------
export PYTHONPATH="$REPO/src"; cd "$REPO"
trc=1
for attempt in 1 2 3; do
  MP=$(( 20000 + RANDOM % 20000 )); echo "--- train attempt $attempt (port $MP) $(date '+%T') ---"
  torchrun --nproc_per_node="$NGPU" --master_port="$MP" \
    src/scripts/train/sft/singletask_ladder/Qwen3-4B-singletask-ladder-SFT-local.py \
    --variant dense --task "$TASK" --data-root "$DATA_ROOT" \
    --seq-len "$SEQ_LEN" --max-steps "$MAX_STEPS" --lr "$LR" --batch-tokens "$BATCH_TOKENS" \
    --save-interval "$SAVE_INTERVAL" --run-name "$RUN_NAME" --save-folder "$SAVE_FOLDER" --wandb-group "$WBGROUP"
  trc=$?; echo "train attempt $attempt rc=$trc"; [ $trc -eq 0 ] && break; sleep 10
done
[ $trc -ne 0 ] && { echo "TRAIN FAILED"; exit $trc; }

STEP_DIR=$(ls -d "$SAVE_FOLDER"/step* 2>/dev/null | sort -V | tail -1)
echo "latest step dir: $STEP_DIR"
[ -f "$STEP_DIR/model_and_optim/.metadata" ] || { echo "no checkpoint to eval"; exit 2; }

# ---------- EVAL (same node, node-local checkpoint) ----------
export PYTHONPATH="$REPO/src:$CR"; cd "$CR"
OUT="$SAVE_FOLDER/eval/${TASK}_14k.json"; mkdir -p "$(dirname "$OUT")"
COMMON="--model-path $STEP_DIR --out $OUT --tokenizer Qwen/Qwen3-4B --max-length $SEQ_LEN --max-test-samples $MAX_TEST --batch-size 8"
case "$TASK" in
  contra)  ARGS="$COMMON --skip-ruler --skip-gen --nq-data /nonexistent_skip_nq --contra-data data/contradiction_eval_pubmed_both_n100_k3.jsonl --contra-max-new-tokens 512" ;;
  nq)      ARGS="$COMMON --skip-ruler --skip-gen --ladder --ladder-tasks nq --ladder-rungs 3k --nq-data data/n2ified_eval_nq_q50.jsonl" ;;
  rerank)  ARGS="$COMMON --skip-ruler --skip-gen --ladder --ladder-tasks rerank --ladder-rungs 2k --rerank-data data/msmarco_dev_rerank_k50_1000.jsonl" ;;
  outlier) ARGS="$COMMON --skip-ruler --skip-gen --ladder --ladder-tasks outlier --ladder-rungs 3k --outlier-data $OUTLIER_EVAL_DATA" ;;
  oolong)  ARGS="$COMMON --skip-ruler --skip-gen --ladder --ladder-tasks oolong --ladder-rungs 8k" ;;
  *) echo "ERROR: unknown TASK=$TASK"; exit 2 ;;
esac
echo "=== EVAL $(date '+%T') -> $OUT ==="
EP=$(( 20000 + RANDOM % 20000 ))
torchrun --nproc_per_node="$NGPU" --master_port="$EP" scripts/eval/eval_lc_native.py $ARGS
erc=$?
mkdir -p "$RESULTS_SHARED"
[ -f "$OUT" ] && { cp "$OUT" "$RESULTS_SHARED/${RUN_NAME}_${TASK}_14k.json"; echo "--- result ---"; cat "$OUT"; }
echo "=== DONE train_rc=$trc eval_rc=$erc result=$RESULTS_SHARED/${RUN_NAME}_${TASK}_14k.json $(date '+%F %T') ==="
