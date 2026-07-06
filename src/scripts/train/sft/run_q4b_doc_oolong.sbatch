#!/bin/bash
# Document-chunked OOLONG SFT (dense or landmark) on ONE jsteinhardt H200 node, then optional NATIVE
# eval (custom attention can't use HF/vLLM). Checkpoints go to node-local /data; eval runs on the
# same node so the step dir is visible.
#
#   # dense smoke (10 steps, no eval):
#   sbatch --job-name=dd-smoke --export=ALL,VARIANT=dense,MAX_STEPS=10,EVAL=0,WANDB=0 \
#     src/scripts/train/sft/run_q4b_doc_oolong.sbatch
#   # landmark full + native eval:
#   sbatch --job-name=dl-oolong --export=ALL,VARIANT=landmark,EVAL=1 \
#     src/scripts/train/sft/run_q4b_doc_oolong.sbatch
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive
#SBATCH --gres=gpu:H200:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=06:00:00
#SBATCH --output=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/.docool_%x_%j.log

set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
CR=/scratch/users/prasann/corpus-reasoning
NGPU="${NGPU:-8}"
VARIANT="${VARIANT:-dense}"                       # dense | landmark
LR="${LR:-5e-5}"
SEQ_LEN="${SEQ_LEN:-4096}"
EPOCHS="${EPOCHS:-3}"
MAX_STEPS="${MAX_STEPS:-0}"                        # >0 = smoke (overrides epochs)
EVAL="${EVAL:-1}"                                  # 1 = run native OOLONG eval after training
WANDB="${WANDB:-1}"
NSAMP="${NSAMP:-100}"
CROSS_DOC_MODE="${CROSS_DOC_MODE:-chunked}"        # dense variant: chunked | hierarchical_dilated
DILATION_N="${DILATION_N:-4}"
DILATION_M="${DILATION_M:-2}"
BASE_CKPT="${BASE_CKPT:-}"                          # override base CPT model_and_optim dir (else launcher default)
RUN_NAME="${RUN_NAME:-q4b-doc${VARIANT}-oolong-ctx2048}"
SAVE_FOLDER=/data/prasann/doc_oolong_runs/$RUN_NAME
EVAL_DATA="${EVAL_DATA:-$CR/data/oolong_test_synth_ctx2048_spliteval.jsonl}"
RESULT=$CR/outputs/eval_results/${RUN_NAME}_docnative.json
echo "=== HOST=$(hostname) RUN=$RUN_NAME VARIANT=$VARIANT LR=$LR SEQ=$SEQ_LEN NGPU=$NGPU EVAL=$EVAL START=$(date '+%F %T') ==="

source /system/linux/miniforge-3.13/etc/profile.d/conda.sh
OLMOENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$OLMOENV" ] || OLMOENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo

export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=lo
# wandb needs the API key in the env (olmo-core reads WANDB_API_KEY, not ~/.netrc).
if [ "$WANDB" = "1" ]; then
  export WANDB_API_KEY=$(awk '/api.wandb.ai/{getline; if($1=="login")getline; for(i=1;i<=NF;i++) if($i=="password") print $(i+1)}' ~/.netrc 2>/dev/null | head -1)
  [ -z "${WANDB_API_KEY:-}" ] && export WANDB_API_KEY=$(grep -A2 'machine api.wandb.ai' ~/.netrc | awk '/password/{print $2}' | head -1)
fi

# Trust SLURM's GPU allocation (nvidia-smi is not cgroup-isolated here).
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  mapfile -t CLEAN < <(nvidia-smi --query-gpu=memory.used,uuid --format=csv,noheader,nounits | awk -F', ' '$1+0 < 2000 {print $2}')
  export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${CLEAN[*]:0:$NGPU}")
fi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# ---------- TRAIN ----------
conda activate "$OLMOENV"
export PYTHONPATH="$REPO/src"
cd "$REPO"
DUR="--epochs $EPOCHS"; [ "$MAX_STEPS" -gt 0 ] && DUR="--max-steps $MAX_STEPS"
WBARG=""; [ "$WANDB" != "1" ] && WBARG="--no-wandb"
# dense-only chunked-attention pattern (eval rebuilds it from the saved config.json, so no eval arg).
CDMARG=""
if [ "$VARIANT" = "dense" ]; then
  CDMARG="--cross-doc-mode $CROSS_DOC_MODE"
  [ "$CROSS_DOC_MODE" = "hierarchical_dilated" ] && CDMARG="$CDMARG --dilation-n $DILATION_N --dilation-m $DILATION_M"
fi
BASEARG=""; [ -n "$BASE_CKPT" ] && BASEARG="--base-ckpt $BASE_CKPT"
echo "=== [1/2] TRAIN $(date '+%T') ==="
trc=1
for attempt in 1 2 3; do
  MASTER_PORT=$(( 20000 + RANDOM % 20000 ))
  echo "--- train attempt $attempt (port $MASTER_PORT) $(date '+%T') ---"
  torchrun --nproc_per_node="$NGPU" --master_port="$MASTER_PORT" \
    src/scripts/train/sft/Qwen3-4B-document-chunked-longctx-SFT-local.py \
    --variant "$VARIANT" --run-name "$RUN_NAME" --save-folder "$SAVE_FOLDER" \
    --lr "$LR" --seq-len "$SEQ_LEN" $DUR $WBARG $CDMARG $BASEARG
  trc=$?; echo "train attempt $attempt rc=$trc"
  [ $trc -eq 0 ] && break
  echo "  (retrying after failure)"; sleep 10
done
[ $trc -ne 0 ] && { echo "TRAIN FAILED after retries"; exit $trc; }

STEP=$(ls -d "$SAVE_FOLDER"/step* 2>/dev/null | sed 's#.*/step##' | sort -n | tail -1)
STEP_DIR="$SAVE_FOLDER/step${STEP}"
echo "latest step dir: $STEP_DIR"
[ -f "$STEP_DIR/model_and_optim/.metadata" ] || { echo "no final checkpoint"; exit 2; }

# ---------- NATIVE EVAL ----------
if [ "$EVAL" = "1" ]; then
  echo "=== [2/2] NATIVE EVAL $(date '+%T') ==="
  export PYTHONPATH="$REPO/src:$CR"   # olmo-core src (with document_chunk_landmark) shadows the editable install
  cd "$CR"
  EVAL_PORT=$(( 20000 + RANDOM % 20000 ))
  torchrun --nproc_per_node="$NGPU" --master_port="$EVAL_PORT" \
    scripts/eval/eval_lc_native_docchunk.py \
    --variant "$VARIANT" --model-path "$STEP_DIR" --out "$RESULT" --tokenizer Qwen/Qwen3-4B \
    --oolong-data "$EVAL_DATA" --max-test-samples "$NSAMP" --mem-freq 63
  echo "eval rc=$? result=$RESULT"
fi
echo "=== DONE $(date '+%T') ==="
