#!/bin/bash
# NON-CHUNKED FAST-LANDMARK Qwen3-4B contradiction-only SFT baseline (n=250), local jsteinhardt H200.
# Plain landmark over the WHOLE sequence (no document chunking). Pure contradiction SFT
# (cpt-frac 0, all task fracs 0 -> contradiction = 100%); fast_landmark=True, mem_freq=63.
#   1) train fast-landmark Qwen3-4B (landmark packing) -> ckpt on node-local /data
#   2) native landmark eval on the n250 contradiction EVAL set -> F1 JSON on shared NFS
# Landmark attention cannot run under HF/vLLM -> native generation module inserts landmark tokens
# at prefill. The launcher writes config.json into each step dir for the native eval.
#
#   sbatch --job-name=contra250-lm --nodelist=cubbins \
#     src/scripts/train/sft/run_q4b_contra_n250_landmark.sbatch
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive
#SBATCH --gres=gpu:H200:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=06:00:00
#SBATCH --output=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/.contra250_%x_%j.log

set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
CR=/scratch/users/prasann/corpus-reasoning
NGPU="${NGPU:-8}"
MODEL="${MODEL:-qwen3_4B}"
LR="${LR:-1e-5}"
SEQ_LEN="${SEQ_LEN:-16384}"                    # multiple of block_size=64 (mem_freq+1); keeps all n250
MEM_FREQ="${MEM_FREQ:-63}"
# Landmark packing pads to ~1 doc/16k-window, so a token budget buys fewer real doc-passes than the
# dense ConcatAndChunk run; bump to ~64M so each n250 example is seen ~2x (matches the dense ~1.9 ep).
TARGET_TOKENS="${TARGET_TOKENS:-64000000}"
MAX_STEPS="${MAX_STEPS:-0}"
RUN_NAME="${RUN_NAME:-q4b-contra250-landmark}"
SFT_DATA_ROOT="${SFT_DATA_ROOT:-/scratch/users/prasann/longctx_sft_qwen/contradiction_n250}"
CONTRA_EVAL="${CONTRA_EVAL:-data/contradiction_eval_pubmed_both_n250_k3.jsonl}"
MAX_SAMPLES="${MAX_SAMPLES:-500}"  # default 500 (uses all 488 of the n250 eval set) for tighter CIs
CONTRA_MAXNEW="${CONTRA_MAXNEW:-2400}"
MEM_ID="${MEM_ID:-151860}"
# Optional overrides for compressive-landmark and early-CPT-base experiments:
#   BASE_CKPT   = model_and_optim dir to init from (default: launcher's fast-landmark step2385)
#   COMPRESSIVE = 1 -> fast_compressive_landmark attention
#   BASE_CONFIG = config.json matching BASE_CKPT's architecture (used as eval config fallback)
BASE_CKPT="${BASE_CKPT:-}"
COMPRESSIVE="${COMPRESSIVE:-0}"
SAVE_FOLDER=/data/prasann/cpt_mix_runs/$RUN_NAME
BASE_CONFIG="${BASE_CONFIG:-/scratch/users/prasann/cpt_mix_ckpts/q4b-fast-landmark-step2385/config.json}"
RESULT=$CR/outputs/eval_results/${RUN_NAME}_lmnative.json
echo "=== HOST=$(hostname) RUN=$RUN_NAME SEQ=$SEQ_LEN LR=$LR TARGET_TOKENS=$TARGET_TOKENS NGPU=$NGPU START=$(date '+%F %T') ==="

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
# fast_landmark + landmark packing + native landmark generation live in the working tree's olmo_core,
# NOT the env's editable clone -> put $REPO/src first; $CR for scripts.eval.* at eval time.
export PYTHONPATH="$REPO/src:$CR"
cd "$REPO"
MAXARG="--target-tokens $TARGET_TOKENS"; [ "$MAX_STEPS" -gt 0 ] && MAXARG="--max-steps $MAX_STEPS --target-tokens 0"
EXTRA=""; [ -n "$BASE_CKPT" ] && EXTRA="$EXTRA --base-ckpt $BASE_CKPT"; [ "$COMPRESSIVE" = "1" ] && EXTRA="$EXTRA --compressive"; [ "${NO_PACK:-0}" = "1" ] && EXTRA="$EXTRA --no-pack"
echo "EXTRA_ARGS='$EXTRA' BASE_CONFIG=$BASE_CONFIG"

# ---------- 1) TRAIN ----------
if [ "${SKIP_TRAIN:-0}" = "1" ]; then
  echo "=== [1/2] TRAIN skipped ==="; trc=0
else
  echo "=== [1/2] TRAIN $(date '+%T') ==="
  trc=1
  for attempt in 1 2 3; do
    MASTER_PORT=$(( 20000 + RANDOM % 20000 ))
    echo "--- train attempt $attempt (port $MASTER_PORT) $(date '+%T') ---"
    torchrun --nproc_per_node="$NGPU" --master_port="$MASTER_PORT" \
      src/scripts/train/sft/Qwen3-4B-fast-landmark-cptmix-5task-local.py \
      --run-name "$RUN_NAME" --save-folder "$SAVE_FOLDER" --model "$MODEL" \
      --sft-data-root "$SFT_DATA_ROOT" \
      --cpt-frac "${CPT_FRAC:-0}" --tulu-frac 0 --nq-frac 0 --oolong-frac 0 --rerank-frac 0 --outlier-frac 0 \
      --lr "$LR" --seq-len "$SEQ_LEN" --mem-freq "$MEM_FREQ" $MAXARG $EXTRA --no-wandb
    trc=$?; echo "train attempt $attempt rc=$trc"
    [ $trc -eq 0 ] && break
    echo "  (retrying after failure)"; sleep 10
  done
fi
[ $trc -ne 0 ] && { echo "TRAIN FAILED after retries"; exit $trc; }

STEP_DIR=$(ls -d "$SAVE_FOLDER"/step* 2>/dev/null | sed 's#.*/step##' | sort -n | tail -1)
STEP_DIR="$SAVE_FOLDER/step${STEP_DIR}"
echo "latest step dir: $STEP_DIR"
[ -f "$STEP_DIR/model_and_optim/.metadata" ] || { echo "no final checkpoint"; exit 2; }
if [ ! -f "$STEP_DIR/config.json" ]; then
  echo "config.json missing; copying from landmark base $BASE_CONFIG"
  cp "$BASE_CONFIG" "$STEP_DIR/config.json" || { echo "could not copy base config.json"; exit 2; }
fi

# ---------- 2) NATIVE LANDMARK EVAL (contradiction only, 8-way DP) ----------
echo "=== [2/2] EVAL (native landmark) $(date '+%T') ==="
cd "$CR"
MASTER_PORT=$(( 20000 + RANDOM % 20000 ))
torchrun --nproc_per_node="$NGPU" --master_port="$MASTER_PORT" \
  scripts/eval/eval_lc_native_landmark.py \
  --model-path "$STEP_DIR" --out "$RESULT" --tokenizer "${TOKENIZER:-Qwen/Qwen3-4B}" \
  --skip-ruler --skip-extra --skip-gen --nq-data /nonexistent --contra-data "$CONTRA_EVAL" \
  --contra-max-new-tokens "$CONTRA_MAXNEW" --max-length "${MAX_LENGTH:-16384}" \
  --max-test-samples "$MAX_SAMPLES" --mem-id "$MEM_ID"
erc=$?
echo "=== DONE $(date '+%T') rc=$erc result=$RESULT ==="
exit $erc
