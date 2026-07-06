#!/bin/bash
# Self-contained FAST-LANDMARK CPT-mix 5-task job for ONE config on ONE jsteinhardt H200 node:
#   1) train fast-landmark Qwen3-4B (5 SFT tasks + CPT mix, landmark packing) -> ckpt on node-local /data
#   2) eval NATIVELY on the olmo-core distcp (RULER + contra + nq + oolong + rerank + outlier + gen)
#      -- landmark attention cannot run under HF/vLLM, so there is NO olmo->HF export here; the native
#      generation module inserts landmark tokens at prefill. Summary JSON -> shared NFS.
#
#   sbatch --job-name=lm-5wF-c85 --export=ALL,RUN_NAME=q4b-lm-5wF-c85 \
#     src/scripts/train/sft/run_q4b_lm_cptmix_5task.sbatch
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=mooney,sneetches
#SBATCH --gres=gpu:H200:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH --time=08:00:00
#SBATCH --output=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/.lmsweep_%x_%j.log

set -uo pipefail
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
CR=/scratch/users/prasann/corpus-reasoning
NGPU="${NGPU:-4}"
MODEL="${MODEL:-qwen3_4B}"
# Default = the WINNING dense 5wF-c85 mix (contra 2x, rerank/outlier 1.5x, nq/oolong 1x of the 0.15
# SFT budget; CPT 0.85). Matched to the dense baseline RULER 0.799 for a clean landmark-vs-dense comp.
CPT_FRAC="${CPT_FRAC:-0.85}"
TULU_FRAC="${TULU_FRAC:-0.0}"
NQ_FRAC="${NQ_FRAC:-0.0214}"
OOLONG_FRAC="${OOLONG_FRAC:-0.0214}"
RERANK_FRAC="${RERANK_FRAC:-0.0321}"
OUTLIER_FRAC="${OUTLIER_FRAC:-0.0321}"
LR="${LR:-1e-5}"
SEQ_LEN="${SEQ_LEN:-8192}"
MEM_FREQ="${MEM_FREQ:-63}"
TARGET_TOKENS="${TARGET_TOKENS:-48000000}"
BATCH_TOKENS="${BATCH_TOKENS:-0}"
MAX_STEPS="${MAX_STEPS:-0}"
RUN_NAME="${RUN_NAME:-q4b-lm-5wF-c85}"
SAVE_FOLDER=/data/prasann/cpt_mix_runs/$RUN_NAME
RESULT=$CR/outputs/eval_results/${RUN_NAME}_lmnative.json
echo "=== HOST=$(hostname) RUN=$RUN_NAME CPT_FRAC=$CPT_FRAC LR=$LR TARGET_TOKENS=$TARGET_TOKENS NGPU=$NGPU START=$(date '+%F %T') ==="

source /system/linux/miniforge-3.13/etc/profile.d/conda.sh
OLMOENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$OLMOENV" ] || OLMOENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo

export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 NCCL_DEBUG=WARN NCCL_SOCKET_IFNAME=lo

# Trust SLURM's GPU allocation (CUDA_VISIBLE_DEVICES per job); only scan if unset.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  mapfile -t CLEAN < <(nvidia-smi --query-gpu=memory.used,uuid --format=csv,noheader,nounits | awk -F', ' '$1+0 < 2000 {print $2}')
  export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${CLEAN[*]:0:$NGPU}")
fi
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

conda activate "$OLMOENV"
# fast_landmark + landmark packing + the native landmark generation module live in the working tree's
# olmo_core, NOT the env's editable clone -> put $REPO/src first; $CR for scripts.eval.* at eval time.
export PYTHONPATH="$REPO/src:$CR"
cd "$REPO"
MAXARG="--target-tokens $TARGET_TOKENS"; [ "$MAX_STEPS" -gt 0 ] && MAXARG="--max-steps $MAX_STEPS --target-tokens 0"

# ---------- 1) TRAIN ----------
BASE_CONFIG=/scratch/users/prasann/cpt_mix_ckpts/q4b-fast-landmark-step2385/config.json
if [ "${SKIP_TRAIN:-0}" = "1" ]; then
  echo "=== [1/2] TRAIN skipped (SKIP_TRAIN=1) $(date '+%T') ==="
  trc=0
else
  echo "=== [1/2] TRAIN $(date '+%T') ==="
  trc=1
  for attempt in 1 2 3; do
    MASTER_PORT=$(( 20000 + RANDOM % 20000 ))
    echo "--- train attempt $attempt (port $MASTER_PORT) $(date '+%T') ---"
    torchrun --nproc_per_node="$NGPU" --master_port="$MASTER_PORT" \
      src/scripts/train/sft/Qwen3-4B-fast-landmark-cptmix-5task-local.py \
      --run-name "$RUN_NAME" --save-folder "$SAVE_FOLDER" --model "$MODEL" \
      --cpt-frac "$CPT_FRAC" --tulu-frac "$TULU_FRAC" --nq-frac "$NQ_FRAC" --oolong-frac "$OOLONG_FRAC" \
      --rerank-frac "$RERANK_FRAC" --outlier-frac "$OUTLIER_FRAC" --lr "$LR" --seq-len "$SEQ_LEN" \
      --mem-freq "$MEM_FREQ" --batch-tokens "$BATCH_TOKENS" $MAXARG --no-wandb
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
# The native eval needs config.json (model + tokenizer) in the step dir. The launcher now writes it,
# but older checkpoints / older launcher versions may lack it -- fall back to the base ckpt's
# config.json (identical model arch + tokenizer; SFT does not change the architecture).
if [ ! -f "$STEP_DIR/config.json" ]; then
  echo "config.json missing in step dir; copying from base ckpt $BASE_CONFIG"
  cp "$BASE_CONFIG" "$STEP_DIR/config.json" || { echo "could not copy base config.json"; exit 2; }
fi

# ---------- 2) NATIVE LANDMARK EVAL (no HF export) ----------
echo "=== [2/2] EVAL (native landmark) $(date '+%T') ==="
cd "$CR"
python -u scripts/eval/eval_lc_native_landmark.py \
  --model-path "$STEP_DIR" --out "$RESULT" --tokenizer "${TOKENIZER:-Qwen/Qwen3-4B}" \
  --max-test-samples "${MAX_SAMPLES:-100}" --ruler-lengths "${RULER_LENGTHS:-L1024,L2048}" \
  --max-length "${MAX_LENGTH:-16384}" --mem-id "${MEM_ID:-151860}" ${EVAL_EXTRA:-}
erc=$?
echo "=== DONE $(date '+%T') rc=$erc result=$RESULT ==="
exit $erc
