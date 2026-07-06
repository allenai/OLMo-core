#!/bin/bash
# Document-chunked LANDMARK contradiction SFT (one landmark ~ one claim) on ONE jsteinhardt H200 node,
# then NATIVE contradiction eval (custom attention can't use HF/vLLM). Direct test of whether the
# document-chunked landmark layout closes the landmark<->standard contradiction gap.
# Checkpoints -> node-local /data; eval runs on the same node so the step dir is visible.
#
#   # n100 (seq 8192):
#   sbatch --job-name=dlc100 --export=ALL,NSET=100,SEQ_LEN=8192,MAX_LENGTH=12288,MAX_SAMPLES=100 \
#     src/scripts/train/sft/run_q4b_docchunk_contra.sbatch
#   # n250 (seq 16384):
#   sbatch --job-name=dlc250 --export=ALL,NSET=250,SEQ_LEN=16384,MAX_LENGTH=20480,MAX_SAMPLES=200 \
#     src/scripts/train/sft/run_q4b_docchunk_contra.sbatch
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive
#SBATCH --gres=gpu:H200:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --output=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/.dlcontra_%x_%j.log

set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
CR=/scratch/users/prasann/corpus-reasoning
NGPU="${NGPU:-8}"
VARIANT="${VARIANT:-landmark}"
LR="${LR:-5e-5}"
NSET="${NSET:-100}"                                # 100 | 250
SEQ_LEN="${SEQ_LEN:-8192}"                         # multiple of block_size=64; 8192 (n100) / 16384 (n250)
EPOCHS="${EPOCHS:-3}"
MAX_STEPS="${MAX_STEPS:-0}"
EVAL="${EVAL:-1}"
WANDB="${WANDB:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"                  # eval samples (n100->100, n250->200)
MAX_LENGTH="${MAX_LENGTH:-12288}"                  # prefill cap = MAX_LENGTH - CONTRA_MAXNEW
CONTRA_MAXNEW="${CONTRA_MAXNEW:-2400}"
RUN_NAME="${RUN_NAME:-q4b-docchunk-lm-contra${NSET}}"
SFT_DATA_ROOT="${SFT_DATA_ROOT:-/scratch/users/prasann/longctx_sft_qwen/contradiction_n${NSET}_doclandmark}"
# base CPT ckpt was renamed to the '-modelonly' convention on 2026-06-26 (old path deleted).
BASE_CKPT="${BASE_CKPT:-/scratch/users/prasann/cpt_mix_ckpts/q4b-fast-landmark-step2385-modelonly/model_and_optim}"
CONTRA_EVAL="${CONTRA_EVAL:-data/contradiction_eval_pubmed_both_n${NSET}_k3.jsonl}"
SAVE_FOLDER=/data/prasann/doc_contra_runs/$RUN_NAME
RESULT=$CR/outputs/eval_results/${RUN_NAME}_docnative.json
echo "=== HOST=$(hostname) RUN=$RUN_NAME VARIANT=$VARIANT NSET=$NSET SEQ=$SEQ_LEN LR=$LR NGPU=$NGPU EVAL=$EVAL START=$(date '+%F %T') ==="

source /system/linux/miniforge-3.13/etc/profile.d/conda.sh
OLMOENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$OLMOENV" ] || OLMOENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo

export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=lo
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
# working-tree olmo_core (document_chunk_landmark + landmark kernel) must shadow the editable install.
export PYTHONPATH="$REPO/src"
cd "$REPO"
DUR="--epochs $EPOCHS"; [ "$MAX_STEPS" -gt 0 ] && DUR="--max-steps $MAX_STEPS"
WBARG=""; [ "$WANDB" != "1" ] && WBARG="--no-wandb"
echo "=== [1/2] TRAIN $(date '+%T') data=$SFT_DATA_ROOT ==="
trc=1
for attempt in 1 2 3; do
  MASTER_PORT=$(( 20000 + RANDOM % 20000 ))
  echo "--- train attempt $attempt (port $MASTER_PORT) $(date '+%T') ---"
  torchrun --nproc_per_node="$NGPU" --master_port="$MASTER_PORT" \
    src/scripts/train/sft/Qwen3-4B-document-chunked-longctx-SFT-local.py \
    --variant "$VARIANT" --run-name "$RUN_NAME" --save-folder "$SAVE_FOLDER" \
    --data-root "$SFT_DATA_ROOT" --base-ckpt "$BASE_CKPT" \
    --lr "$LR" --seq-len "$SEQ_LEN" $DUR $WBARG
  trc=$?; echo "train attempt $attempt rc=$trc"
  [ $trc -eq 0 ] && break
  echo "  (retrying after failure)"; sleep 10
done
[ $trc -ne 0 ] && { echo "TRAIN FAILED after retries"; exit $trc; }

STEP=$(ls -d "$SAVE_FOLDER"/step* 2>/dev/null | sed 's#.*/step##' | sort -n | tail -1)
STEP_DIR="$SAVE_FOLDER/step${STEP}"
echo "latest step dir: $STEP_DIR"
[ -f "$STEP_DIR/model_and_optim/.metadata" ] || { echo "no final checkpoint"; exit 2; }

# ---------- NATIVE CONTRADICTION EVAL ----------
if [ "$EVAL" = "1" ]; then
  echo "=== [2/2] NATIVE CONTRA EVAL $(date '+%T') ==="
  export PYTHONPATH="$REPO/src:$CR"
  cd "$CR"
  EVAL_PORT=$(( 20000 + RANDOM % 20000 ))
  torchrun --nproc_per_node="$NGPU" --master_port="$EVAL_PORT" \
    scripts/eval/eval_lc_native_docchunk_contra.py \
    --variant "$VARIANT" --model-path "$STEP_DIR" --out "$RESULT" --tokenizer Qwen/Qwen3-4B \
    --contra-data "$CONTRA_EVAL" --max-test-samples "$MAX_SAMPLES" \
    --contra-max-new-tokens "$CONTRA_MAXNEW" --max-length "$MAX_LENGTH" --mem-freq 63
  echo "eval rc=$? result=$RESULT"
  echo "--- RESULT ---"; cat "$RESULT" 2>/dev/null; echo
fi
echo "=== DONE $(date '+%T') ==="
