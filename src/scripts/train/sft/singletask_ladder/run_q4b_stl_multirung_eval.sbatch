#!/bin/bash
# MULTI-RUNG native eval for the overnight 10k matrix checkpoints, on ONE H200 node. Native olmo_core
# generate (NO HF/vLLM) so it works for dense / compressive / landmark / docchunk. Reads the larger
# (_600/_500) per-rung eval files from /scratch/users/prasann/cpt_data/eval500 via the LADDERS map in
# scripts/eval/eval_lc_native.py; rerank uses the CE-graded eval files (NDCG@10 + Kendall-tau).
#
# NOTE: eval DATA (cpt_data/eval500, corpus-reasoning/data/*.jsonl, the CE files) lives on LOCAL
# /scratch + the corpus-reasoning checkout, NOT on weka/Beaker -- so this runs LOCALLY after the Beaker
# checkpoint is synced down to a local step dir. (To fold eval into the Beaker job instead, upload all
# eval data to weka and init the corpus-reasoning submodule on the node -- see the overnight report.)
#
#   sbatch --job-name=ev-contra-dense \
#     --export=ALL,TASK=contra,VARIANT=dense,CKPT=/data/prasann/.../q4b-dense-contra-ladder32k-10k/stepNNN \
#     src/scripts/train/sft/singletask_ladder/run_q4b_stl_multirung_eval.sbatch
#
#   TASK    = contra | nq | rerank | outlier | oolong
#   VARIANT = dense | compressive | landmark | docchunk   (docchunk currently evals OOLONG only)
#   CKPT    = a STEP DIR (config.json + model_and_optim/.metadata)
#   RUNGS   = comma rung list (default per task below); MAX_TEST (default 600); MAX_LENGTH (default 40960)
#
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive
#SBATCH --gres=gpu:H200:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=03:00:00
#SBATCH --output=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/.mreval_%x_%j.log
set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
CR=/scratch/users/prasann/corpus-reasoning
NGPU="${NGPU:-8}"
TASK="${TASK:?set TASK=contra|nq|rerank|outlier|oolong}"
VARIANT="${VARIANT:?set VARIANT=dense|compressive|landmark|docchunk}"
CKPT="${CKPT:?set CKPT=<step dir with config.json + model_and_optim/>}"
MAX_TEST="${MAX_TEST:-600}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_LENGTH="${MAX_LENGTH:-40960}"
TOKENIZER="${TOKENIZER:-Qwen/Qwen3-4B}"
CKPT="${CKPT%/}"; RUN_DIR="$(dirname "$CKPT")"; RUN="$(basename "$RUN_DIR")"
RESULTS_SHARED=/scratch/users/prasann/stl_eval_results
OUT="$RUN_DIR/eval/${TASK}_multirung.json"; mkdir -p "$(dirname "$OUT")" "$RESULTS_SHARED"

# Default rungs per task (contra/rerank base rung is 2k, not 3k; the 8k/16k/32k files are _600/_500).
case "$TASK" in
  contra)  RUNGS="${RUNGS:-2k,8k,16k,32k}" ;;
  nq)      RUNGS="${RUNGS:-3k,8k,16k,32k}" ;;
  outlier) RUNGS="${RUNGS:-3k,8k,16k,32k}" ;;
  oolong)  RUNGS="${RUNGS:-8k,16k,32k}" ;;
  rerank)  RUNGS="${RUNGS:-ce}" ;;   # rerank handled via the CE-graded files, below
  *) echo "ERROR unknown TASK=$TASK"; exit 2 ;;
esac
echo "=== HOST=$(hostname) RUN=$RUN TASK=$TASK VARIANT=$VARIANT RUNGS=$RUNGS CKPT=$CKPT START=$(date '+%F %T') ==="
[ -f "$CKPT/model_and_optim/.metadata" ] || { echo "ERROR: $CKPT not a complete step dir"; exit 2; }

source /system/linux/miniforge-3.13/etc/profile.d/conda.sh
OLMOENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$OLMOENV" ] || OLMOENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=lo
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  mapfile -t CLEAN < <(nvidia-smi --query-gpu=memory.used,uuid --format=csv,noheader,nounits | awk -F', ' '$1+0 < 2000 {print $2}')
  export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${CLEAN[*]:0:$NGPU}")
fi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
conda activate "$OLMOENV"
export PYTHONPATH="$REPO/src:$CR"   # olmo-core src (custom attn) shadows the editable install
cd "$CR"
PORT=$(( 20000 + RANDOM % 20000 ))

# ---- docchunk: custom box-marker prefill eval; only OOLONG is wired in eval_lc_native_docchunk.py ----
if [ "$VARIANT" = "docchunk" ]; then
  if [ "$TASK" != "oolong" ]; then
    echo "NOTE: docchunk native eval currently supports OOLONG only (eval_lc_native_docchunk.py)."
    echo "      $TASK docchunk eval needs a per-task box-marker eval path -- skipping."; exit 0
  fi
  for rung in 1024 2048 4096 8192 16384 32768; do
    DF="$CR/data/oolong_test_synth_ctx${rung}_spliteval.jsonl"
    [ -f "$DF" ] || continue
    O="$RUN_DIR/eval/oolong_docchunk_${rung}.json"
    echo "=== docchunk oolong ctx$rung -> $O ==="
    torchrun --nproc_per_node="$NGPU" --master_port="$PORT" scripts/eval/eval_lc_native_docchunk.py \
      --variant dense --model-path "$CKPT" --out "$O" --tokenizer "$TOKENIZER" \
      --oolong-data "$DF" --max-test-samples "$MAX_TEST" --max-length "$MAX_LENGTH" --mem-freq 63
    cp "$O" "$RESULTS_SHARED/${RUN}_oolong_${rung}.json" 2>/dev/null
  done
  echo "=== DONE docchunk $(date '+%F %T') ==="; exit 0
fi

# ---- rerank: CE-graded eval files (k20~3k, k50~8k, k100~16k); old LADDERS rerank files raise
#      NotImplementedError, so eval each CE file at the base ladder rung via --rerank-data. ----
if [ "$TASK" = "rerank" ]; then
  declare -A CE=( [3k]=data/msmarco_trainhn_eval_k20_500.jsonl
                  [8k]=data/msmarco_trainhn_eval_k50_500.jsonl
                  [16k]=data/msmarco_trainhn_eval_k100_500.jsonl )
  for r in 3k 8k 16k; do
    CEF="${CE[$r]}"; [ -f "$CR/$CEF" ] || { echo "MISSING $CEF, skipping"; continue; }
    O="$RUN_DIR/eval/rerank_ce_${r}.json"
    echo "=== rerank CE @${r} ($CEF) -> $O ==="
    torchrun --nproc_per_node="$NGPU" --master_port="$PORT" scripts/eval/eval_lc_native.py \
      --model-path "$CKPT" --out "$O" --tokenizer "$TOKENIZER" --max-length "$MAX_LENGTH" \
      --max-test-samples "$MAX_TEST" --batch-size "$BATCH_SIZE" --skip-ruler --skip-gen \
      --ladder --ladder-tasks rerank --ladder-rungs 2k --rerank-data "$CEF"
    cp "$O" "$RESULTS_SHARED/${RUN}_rerank_ce_${r}.json" 2>/dev/null
  done
  echo "=== DONE rerank-CE $(date '+%F %T') ==="; exit 0
fi

# ---- dense / compressive / landmark: standard multi-rung ladder (NDCG/F1/score per rung). ----
COMMON="--model-path $CKPT --out $OUT --tokenizer $TOKENIZER --max-length $MAX_LENGTH \
        --max-test-samples $MAX_TEST --batch-size $BATCH_SIZE --skip-ruler --skip-gen \
        --ladder --ladder-tasks $TASK --ladder-rungs $RUNGS"
case "$TASK" in
  contra)  LTASK=contradiction; EXTRA="--contra-data data/contradiction_eval_pubmed_both_n100_k3.jsonl --contra-max-new-tokens 512" ;;
  nq)      LTASK=nq;            EXTRA="--nq-data data/n2ified_eval_nq_q50.jsonl" ;;
  outlier) LTASK=outlier;       EXTRA="--outlier-data data/outlier_wiki100w_n55_k3_eval_600.jsonl" ;;
  oolong)  LTASK=oolong;        EXTRA="" ;;
esac
# --ladder-tasks uses the loadtask name (contradiction/nq/oolong/outlier).
COMMON="${COMMON/--ladder-tasks $TASK/--ladder-tasks $LTASK}"
echo "=== EVAL $(date '+%T') -> $OUT ==="
torchrun --nproc_per_node="$NGPU" --master_port="$PORT" scripts/eval/eval_lc_native.py $COMMON $EXTRA
rc=$?
[ -f "$OUT" ] && { cp "$OUT" "$RESULTS_SHARED/${RUN}_${TASK}_multirung.json"; echo "--- $OUT ---"; cat "$OUT"; }
echo "=== DONE rc=$rc result=$RESULTS_SHARED/${RUN}_${TASK}_multirung.json $(date '+%F %T') ==="
exit $rc
