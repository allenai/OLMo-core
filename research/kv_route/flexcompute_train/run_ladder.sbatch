#!/bin/bash
# The flexcompute speed/quality ladder: REAL contradiction SFT runs of train_ctc_suite.py
# on Qwen3.5-4B, short-heavy 2k-32k mix, packed to 65536, one arm after another.
#
# Everything except the router flags is copied from the established contradiction SFT recipe
# (the prior taskscale_lengthmix launch_arms.sh + research/ffn_compute/flop_scaling/launch_grid35.py): same base,
# --seq-len 65536 --pack, --lr 5e-6, --global-batch 8 --micro-batch-instances 1, 1 epoch.
# The ONE deliberate departure is NGPU=4 instead of 8, because the 8-GPU/user cap on jsteinhardt
# is shared with a concurrent benchmark job; every arm uses the same 4 GPUs on the same node, so
# arm-to-arm comparisons are unaffected.
#
#   ARMS="dense headgate moderate aggressive aggr-nokv aggr-cap dense2" sbatch \
#     research/kv_route/flexcompute_train/run_ladder.sbatch
#
# Every arm in ARMS runs SEQUENTIALLY inside ONE job on ONE allocation, so no arm-to-arm
# comparison crosses a job boundary. The dense control is run FIRST and LAST (`dense`,
# `dense2`) to bound drift: this hardware drifts up to ~2x within a single process, see
# research/kv_route/flexcompute_40x/README.md.
#SBATCH --job-name=fxtrain
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodelist=cubbins
#SBATCH --nodes=1
#SBATCH --gres=gpu:H200:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=300G
#SBATCH --time=10:00:00
#SBATCH --output=/data/prasann/joblogs/fxtrain_%x_%j.log
set -uo pipefail

REPO_ROOT=${REPO_ROOT:-$(git -C "$(dirname "$0")" rev-parse --show-toplevel 2>/dev/null || git -C "${SLURM_SUBMIT_DIR:-.}" rev-parse --show-toplevel)}
REPO="${REPO:-$REPO_ROOT}"
ARMS="${ARMS:?set ARMS=\"dense headgate moderate aggressive aggr-nokv aggr-cap dense2\"}"
GEN="${GEN:-g1}"
NGPU="${NGPU:-4}"
DATA_ARM="${DATA_ARM:-contradiction_sh16M_q35}"
MAX_STEPS="${MAX_STEPS:-0}"
SEED="${SEED:-0}"
WORK_ROOT=/data/prasann/flexcompute_train
DATA="$WORK_ROOT/shards/$DATA_ARM"
BASE_CKPT=/data/prasann/ctc_suite/bases/q35-4b-base-modelonly/model_and_optim

# Mirror the node-local log to /scratch so it is readable off-node while the job runs.
# /scratch hit 100%% of its 500G quota on 2026-09-09; mirror node-local instead.
MIRROR_DIR=/data/prasann/flexcompute_train/logs
mkdir -p "$MIRROR_DIR" 2>/dev/null || true
_SRC_LOG="/data/prasann/joblogs/fxtrain_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
_DST_LOG="$MIRROR_DIR/${GEN}_${SLURM_JOB_ID}.log"
_mirror() { cp -f "$_SRC_LOG" "$_DST_LOG" 2>/dev/null || true; }
( while true; do _mirror; sleep 20; done ) &
_MIRROR_PID=$!
trap '_mirror; kill $_MIRROR_PID 2>/dev/null' EXIT

ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$ENV" ] || { echo "FATAL: node-local env missing at $ENV"; exit 5; }
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$REPO/src"
export TOKENIZERS_PARALLELISM=false PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TRITON_CACHE_DIR=/data/prasann/triton_cache
export TMPDIR=/data/prasann/tmp
mkdir -p "$TRITON_CACHE_DIR" "$TMPDIR"

if [ -z "${WANDB_API_KEY:-}" ] && [ -r "$HOME/.netrc" ]; then
  WANDB_API_KEY=$(awk '/machine[[:space:]]+api\.wandb\.ai/{f=1} f&&/password/{print $2; exit}' "$HOME/.netrc")
  [ -n "$WANDB_API_KEY" ] && export WANDB_API_KEY
fi
[ -n "${WANDB_API_KEY:-}" ] || { echo "FATAL: WANDB_API_KEY unset"; exit 4; }

[ -f "$BASE_CKPT/.metadata" ] || { echo "FATAL: base $BASE_CKPT has no .metadata"; exit 3; }
[ -s "$DATA/metadata.json" ] || { echo "FATAL: data $DATA missing"; exit 3; }

# FFN "f_null" -> --ffn-moe-target: with the default rung ladder and the measured working-rung
# shape used by research/kv_route/flexcompute_40x/bench_router_matrix.py (WORK_SHAPE), f_null=0.5 is a mean
# FFN cost of 0.098 and f_null=1.0 is 0. Hence target 0.10 / 0.01.
FFN_MOD="--ffn-moe-target 0.10 --ffn-moe-two-sided --ffn-moe-explore 0.0"
FFN_AGG="--ffn-moe-target 0.01 --ffn-moe-two-sided --ffn-moe-explore 0.0"

echo "=== HOST=$(hostname) ARMS='$ARMS' GEN=$GEN NGPU=$NGPU DATA=$DATA_ARM \
MAX_STEPS=$MAX_STEPS START=$(date '+%F %T') ==="
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader
cd "$REPO"

for ARM in $ARMS; do
  RUN="fxt-${GEN}-${ARM}"
  SAVE="$WORK_ROOT/ckpts/$RUN"
  WORKDIR="$WORK_ROOT/cache-${SLURM_JOB_ID}-${ARM}"
  # NO --block-skip-capacity on the trainable arms. Fixed capacity (mixture-of-depths top-k) binds
  # from call 0 and BYPASSES the annealed budget, so the keep-all init no longer reproduces the
  # base model -- block_skip.py says so in as many words. Measured here: a 3-step smoke of the
  # `moderate` settings WITH capacity 0.5 started at CE 41.4 and was still at 8.3 by step 3, i.e.
  # the shock destroys the model before the run has any steps to recover in. The `aggr-cap` arm
  # keeps it, on purpose, to price what the static shape buys and what it costs.
  case "$ARM" in
    dense|dense2) VARIANT=full; EXTRA="" ;;
    headgate)     VARIANT=full; EXTRA="--sparse-loss-positions" ;;
    moderate)     VARIANT=flexcompute
                  EXTRA="--sparse-loss-positions $FFN_MOD --kv-route-target 0.50 \
                         --block-skip-target 0.50 --block-skip-compact" ;;
    aggressive)   VARIANT=flexcompute
                  EXTRA="--sparse-loss-positions $FFN_AGG --kv-route-target 0.25 \
                         --block-skip-target 0.125 --block-skip-compact" ;;
    aggr-nokv)    VARIANT=flexcompute
                  EXTRA="--sparse-loss-positions $FFN_AGG --kv-route-off \
                         --block-skip-target 0.125 --block-skip-compact" ;;
    # ---- the block-skip KEEP sweep, and why it uses --block-skip-capacity ----
    # The budget-hinge route does NOT reach an operating point in a run this length. The router is
    # keep-all initialised at bias +10, where d sigmoid/d logit = 4.5e-5, and under Adam the bias
    # moves ~router_lr per step, so it needs ~10/1e-3 = 1e4 steps to cross zero. MEASURED on the
    # `moderate` arm (target 0.50, 31 steps): block_skip_keep_frac = 0.99999, kv_route_keep_frac =
    # 1.000 -- the routers charged their overhead and skipped nothing. Fixed capacity is therefore
    # the only mechanism that binds during training; it is also a shock (it ignores the anneal),
    # which is exactly the trade this sweep prices. Same FFN target and no KV router across all
    # four, so the only thing that varies is the capacity.
    capbase)      VARIANT=flexcompute
                  EXTRA="--sparse-loss-positions $FFN_MOD --kv-route-off \
                         --block-skip-target 0.50 --block-skip-compact" ;;
    cap50)        VARIANT=flexcompute
                  EXTRA="--sparse-loss-positions $FFN_MOD --kv-route-off \
                         --block-skip-target 0.50 --block-skip-capacity 0.50 --block-skip-compact" ;;
    cap25)        VARIANT=flexcompute
                  EXTRA="--sparse-loss-positions $FFN_MOD --kv-route-off \
                         --block-skip-target 0.25 --block-skip-capacity 0.25 --block-skip-compact" ;;
    cap125)       VARIANT=flexcompute
                  EXTRA="--sparse-loss-positions $FFN_MOD --kv-route-off \
                         --block-skip-target 0.125 --block-skip-capacity 0.125 --block-skip-compact" ;;
    aggr-cap)     VARIANT=flexcompute
                  EXTRA="--sparse-loss-positions $FFN_AGG --kv-route-target 0.25 \
                         --block-skip-target 0.125 --block-skip-capacity 0.125 --block-skip-compact" ;;
    *) echo "FATAL: unknown ARM=$ARM"; exit 2 ;;
  esac
  # Fresh save folder per arm: the Trainer SILENTLY auto-resumes from a step checkpoint left in
  # the save folder, which would make the step counts and the loss curve garbage.
  if compgen -G "$SAVE/step*" > /dev/null; then
    echo "!!! SKIP $ARM: $SAVE already holds a step checkpoint (would silently resume)"; continue
  fi
  mkdir -p "$SAVE" "$WORKDIR"
  echo "######## ARM=$ARM RUN=$RUN VARIANT=$VARIANT $(date '+%F %T') ########"
  echo "######## EXTRA: $EXTRA"
  MAXARG=""; [ "$MAX_STEPS" -gt 0 ] && MAXARG="--max-steps $MAX_STEPS"
  MASTER_PORT=$((29100 + RANDOM % 800))
  # Bare `torchrun` resolves to the system miniforge on this cluster; go through the node-local
  # interpreter explicitly.
  "$ENV/bin/python" -m torch.distributed.run --nproc_per_node="$NGPU" --master_port="$MASTER_PORT" \
    recipes/train/research_trainer/train_ctc_suite.py \
    --task contradiction --variant "$VARIANT" --model-scale 4b --model-family qwen3_5 \
    --run-name "$RUN" --wandb-group "flexcompute-train-$GEN" \
    --base-checkpoint "$BASE_CKPT" --data "$DATA" --work-dir "$WORKDIR" --save-folder "$SAVE" \
    --seq-len 65536 --pack --epochs 1 --lr 5e-6 \
    --global-batch 8 --micro-batch-instances 1 --seed "$SEED" \
    --save-checkpoint --no-compile --no-final-checkpoint \
    $MAXARG $EXTRA
  echo "######## DONE ARM=$ARM rc=$? $(date '+%F %T') ########"
  rm -rf "$WORKDIR"
done
_mirror
echo "=== ALL ARMS DONE $(date '+%F %T') ==="
