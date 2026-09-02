#!/bin/bash
# Nested-FFN-MoE experiment (contradiction, Qwen3-4B, LOCAL). Standalone FFN-compute study:
# plain causal attention, no KV compaction. FFN_MOE_START picks the arm:
#   FFN_MOE_START=-1 -> dense reference run (no router). The accuracy + wall-clock baseline.
#   FFN_MOE_START=4  -> routed from layer 4 on (the default experimental arm).
#
# TRAP: pass overrides as an env PREFIX on the sbatch line (`BASE=... RUN=... sbatch --export=ALL,BASE,RUN ...`).
# A plain `BASE=...` shell assignment on its own line is NOT exported, `--export=ALL,BASE` then
# passes nothing, and the run silently uses the default base (it cost two all-layer arms a load
# failure on a base that lacked their router keys).
#
# NFS-safe (local_cluster.md): --output on node-local /data + /scratch mirror, node-local env,
# data and base on the node's /data (ffnmoe_exp/). See README.md in this folder.
#
#   sbatch -w sneetches -p jsteinhardt -q preemptive_high --gres=gpu:H200:2 \
#     --export=ALL,RUN=q4b-ffnmoe-dense,FFN_MOE_START=-1 run_q4b_ffnmoe.sbatch
#   sbatch -w sneetches -p jsteinhardt -q preemptive_high --gres=gpu:H200:2 \
#     --export=ALL,RUN=q4b-ffnmoe-v1,TARGET=0.05 run_q4b_ffnmoe.sbatch
#SBATCH --job-name=q4b-ffnmoe
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:H200:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=/data/prasann/joblogs/q4b_ffnmoe_%x_%j.log

set -uo pipefail
MIRROR_DIR=/scratch/users/prasann/ffnmoe_logs
mkdir -p "$MIRROR_DIR" 2>/dev/null || true
_SRC_LOG="/data/prasann/joblogs/q4b_ffnmoe_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
_DST_LOG="$MIRROR_DIR/train_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
_mirror_log() { cp -f "$_SRC_LOG" "$_DST_LOG" 2>/dev/null || true; }
( while true; do _mirror_log; sleep 30; done ) &
_MIRROR_PID=$!
trap '_mirror_log; kill $_MIRROR_PID 2>/dev/null' EXIT

REPO="${REPO:-/accounts/projects/berkeleynlp/prasann/projects/OLMo-core}"
NGPU="${NGPU:-2}"
RUN="${RUN:?set RUN=<run-name>}"
EPOCHS="${EPOCHS:-3}"
SEQ_LEN="${SEQ_LEN:-6144}"
LR="${LR:-1e-5}"
SEED="${SEED:-0}"
# Router arm knobs (see Qwen3-4B-ffnmoe-contra-local.py --help for the rest).
FFN_MOE_START="${FFN_MOE_START:-4}"
DIVISORS="${DIVISORS:-1,4,16,64}"
TARGET="${TARGET:-0.05}"
BUDGET_W="${BUDGET_W:-1.0}"
AC_MODE="${AC_MODE:-none}"   # none (normal training, the default) | full | selected
FRESH="${FRESH:-0}"          # FRESH=1 wipes an existing save dir first (else the trainer RESUMES it)
EXTRA="${EXTRA:-}"
WANDB_GROUP="${WANDB_GROUP:-ffnmoe-contra}"
echo "=== HOST=$(hostname) RUN=$RUN FFN_MOE_START=$FFN_MOE_START TARGET=$TARGET SEQ_LEN=$SEQ_LEN AC_MODE=$AC_MODE START=$(date '+%F %T') ==="

ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$ENV" ] || ENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$REPO/src"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONWARNINGS=ignore
export PYTHONUNBUFFERED=1
if [ -z "${WANDB_API_KEY:-}" ]; then
  WANDB_API_KEY=$(awk '/machine api.wandb.ai/{f=1} f&&/password/{print $2; exit}' "$HOME/.netrc" 2>/dev/null)
  export WANDB_API_KEY
fi
WANDB_FLAG=""
[ -z "${WANDB_API_KEY:-}" ] && WANDB_FLAG="--no-wandb" && echo "WARN: no wandb creds -> --no-wandb"
echo "wandb group: https://wandb.ai/prasann-uc-berkeley-electrical-engineering-computer-sciences/memory-networks/groups/${WANDB_GROUP}"

mapfile -t CLEAN < <(nvidia-smi --query-gpu=memory.used,uuid --format=csv,noheader,nounits | awk -F', ' '$1+0 < 2000 {print $2}')
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${CLEAN[*]:0:$NGPU}")
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# The routed arm MUST load a base with the baked _nffn_* keys (bake_ffn_moe_into_base.py);
# a base without them fails at load. The dense arm can use either.
BASE="${BASE:-/data/prasann/ffnmoe_exp/q4b-dense-cpt-fixmark-ffnmoe/model_and_optim}"
if [ ! -f "$BASE/.metadata" ]; then
  echo "FATAL: base checkpoint $BASE missing .metadata (run bake_ffn_moe_into_base.py first)"; exit 3
fi
DATA="${DATA:-/data/prasann/ffnmoe_exp/contra_n100_qboth_train}"
[ -f "$DATA/metadata.json" ] || { echo "FATAL: data shard $DATA missing"; exit 4; }

cd "$REPO"
SAVE_DIR=/data/prasann/ffnmoe_exp/runs/$RUN
if [ "$FRESH" = "1" ] && [ -d "$SAVE_DIR" ]; then
  echo "FRESH=1: removing existing $SAVE_DIR"; rm -rf "$SAVE_DIR"
fi
# Auto-resume chain. sneetches SIGSEGVs (exitcode -11) roughly once per 300 steps -- it killed a
# 3000-step run at step 830 and another at 1250. The trainer silently resumes from the newest
# checkpoint in the save folder when relaunched under the SAME run name, so retry in-place instead
# of losing the run. Partial "*-tmp" checkpoint dirs from the interrupted save are removed first.
# NOTE: because resume keys off the save folder, never reuse a RUN name for a different config.
MAX_TRIES="${MAX_TRIES:-4}"
RC=1
for TRY in $(seq 1 "$MAX_TRIES"); do
  rm -rf "$SAVE_DIR"/*-tmp 2>/dev/null
  MASTER_PORT=$((29000 + RANDOM % 1000))
  echo "=== attempt $TRY/$MAX_TRIES (port $MASTER_PORT) $(date '+%F %T') ==="
  torchrun --nproc_per_node="$NGPU" --master_port="$MASTER_PORT" \
    src/scripts/train/memexpress/ffnmoe/Qwen3-4B-ffnmoe-contra-local.py \
    --run-name "$RUN" --data "$DATA" --base-ckpt "$BASE" \
    --seq-len "$SEQ_LEN" --epochs "$EPOCHS" --lr "$LR" --seed "$SEED" \
    --ffn-moe-start-layer "$FFN_MOE_START" --ffn-moe-divisors "$DIVISORS" \
    --ffn-moe-target "$TARGET" --ffn-moe-budget-weight "$BUDGET_W" --ac-mode "$AC_MODE" \
    --wandb-group "$WANDB_GROUP" \
    $EXTRA $WANDB_FLAG
  RC=$?
  [ "$RC" -eq 0 ] && break
  # The model-only export at the run root is the completion marker; if it exists the run finished
  # and a nonzero rc came from teardown, not training.
  if [ -f "$SAVE_DIR/model_and_optim/.metadata" ]; then
    echo "=== attempt $TRY rc=$RC but export exists -> treating as complete ==="; RC=0; break
  fi
  echo "=== attempt $TRY FAILED rc=$RC; resuming from newest checkpoint ==="
  sleep 20
done
# Trainer step checkpoints carry ~48GB of fp32 optimizer state each -- only the model-only export
# (model_and_optim + config.json at the run root) is needed for eval.
if [ "$RC" -eq 0 ] && [ -f "$SAVE_DIR/model_and_optim/.metadata" ]; then
  rm -rf "$SAVE_DIR"/step* 2>/dev/null
  echo "cleaned trainer step checkpoints under $SAVE_DIR"
fi
echo "=== DONE $(date '+%F %T') rc=$RC ==="
exit $RC
