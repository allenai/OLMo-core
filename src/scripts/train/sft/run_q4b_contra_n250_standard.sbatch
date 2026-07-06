#!/bin/bash
# NON-CHUNKED STANDARD (full-attention) Qwen3-4B contradiction-only SFT baseline (n=250), local
# jsteinhardt H200. Pure contradiction SFT (cpt-frac 0, all task fracs 0 -> contradiction = 100%).
#   1) train dense Qwen3-4B -> ckpt on node-local /data
#   2) native olmo_core eval on the n250 contradiction EVAL set -> F1 JSON on shared NFS
# Native eval (no HF export): the dense launcher now writes config.json into each step dir.
#
#   sbatch --job-name=contra250-std --nodelist=mooney \
#     src/scripts/train/sft/run_q4b_contra_n250_standard.sbatch
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
SEQ_LEN="${SEQ_LEN:-16384}"
TARGET_TOKENS="${TARGET_TOKENS:-48000000}"   # ~2 epochs of the 100%-contradiction n250 set (~25M tok/epoch)
MAX_STEPS="${MAX_STEPS:-0}"
RUN_NAME="${RUN_NAME:-q4b-contra250-standard}"
SFT_DATA_ROOT="${SFT_DATA_ROOT:-/scratch/users/prasann/longctx_sft_qwen/contradiction_n250}"
CONTRA_EVAL="${CONTRA_EVAL:-data/contradiction_eval_pubmed_both_n250_k3.jsonl}"
MAX_SAMPLES="${MAX_SAMPLES:-500}"  # default 500 (uses all 488 of the n250 eval set) for tighter CIs
CONTRA_MAXNEW="${CONTRA_MAXNEW:-2400}"        # enumerate-CoT answers are ~1960 tok -> need a big budget
SAVE_FOLDER=/data/prasann/cpt_mix_runs/$RUN_NAME
RESULT=$CR/outputs/eval_results/${RUN_NAME}_native.json
echo "=== HOST=$(hostname) RUN=$RUN_NAME SEQ=$SEQ_LEN LR=$LR TARGET_TOKENS=$TARGET_TOKENS NGPU=$NGPU START=$(date '+%F %T') ==="

source /system/linux/miniforge-3.13/etc/profile.d/conda.sh
OLMOENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$OLMOENV" ] || OLMOENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo

export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=lo

# Trust SLURM's per-job GPU allocation; only scan if unset (do NOT node-wide nvidia-smi scan).
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  mapfile -t CLEAN < <(nvidia-smi --query-gpu=memory.used,uuid --format=csv,noheader,nounits | awk -F', ' '$1+0 < 2000 {print $2}')
  export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${CLEAN[*]:0:$NGPU}")
fi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

conda activate "$OLMOENV"
export PYTHONPATH="$REPO/src:$CR"
cd "$REPO"
MAXARG="--target-tokens $TARGET_TOKENS"; [ "$MAX_STEPS" -gt 0 ] && MAXARG="--max-steps $MAX_STEPS --target-tokens 0"
EXTRA=""; [ -n "${BASE_CKPT:-}" ] && EXTRA="--base-ckpt $BASE_CKPT"; echo "EXTRA_ARGS='$EXTRA'"

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
      src/scripts/train/sft/Qwen3-4B-dense-cptmix-contra-local.py \
      --run-name "$RUN_NAME" --save-folder "$SAVE_FOLDER" --model "$MODEL" \
      --sft-data-root "$SFT_DATA_ROOT" \
      --cpt-frac 0 --tulu-frac 0 --nq-frac 0 --oolong-frac 0 --rerank-frac 0 --outlier-frac 0 \
      --lr "$LR" --seq-len "$SEQ_LEN" $MAXARG $EXTRA --no-wandb
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
  echo "config.json missing; copying from dense base"
  cp /scratch/users/prasann/cpt_mix_ckpts/q4b-dense-cpt-step2385-modelonly/config.json "$STEP_DIR/config.json"
fi

# ---------- 2) NATIVE EVAL (contradiction only, 8-way DP) ----------
echo "=== [2/2] EVAL (native dense) $(date '+%T') ==="
cd "$CR"
MASTER_PORT=$(( 20000 + RANDOM % 20000 ))
torchrun --nproc_per_node="$NGPU" --master_port="$MASTER_PORT" \
  scripts/eval/eval_lc_native.py \
  --model-path "$STEP_DIR" --out "$RESULT" --tokenizer "${TOKENIZER:-Qwen/Qwen3-4B}" \
  --skip-ruler --skip-gen --nq-data /nonexistent --contra-data "$CONTRA_EVAL" \
  --contra-max-new-tokens "$CONTRA_MAXNEW" --max-length "${MAX_LENGTH:-16384}" \
  --batch-size "${BATCH_SIZE:-2}" --max-test-samples "$MAX_SAMPLES"
erc=$?
echo "=== DONE $(date '+%T') rc=$erc result=$RESULT ==="
exit $erc
