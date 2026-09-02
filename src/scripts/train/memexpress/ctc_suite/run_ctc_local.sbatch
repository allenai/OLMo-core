#!/bin/bash
# CTC-suite LOCAL SFT launcher (records/ctc-suite-scaling-plan.md): one job = one (task, arm) run
# of train_ctc_suite.py on 8 GPUs of a single Berkeley H200 node. Generalizes
# attn_explore/run_q35_08b_attn_explore.sbatch to any task/variant/scale.
#
# NFS-safe (see local-train-nfs-log-hang memory): --output on node-local /data (mirrored to
# /scratch every 30s + on exit), direct PATH (no conda activate), PYTHONWARNINGS=ignore,
# data+workdir+ckpt staged node-local; the distcp base is read IN PLACE from /scratch (reads are
# NFS-safe; staging via cp can drop the hidden .metadata -> silent train-from-scratch, guarded).
#
# Knobs (env): TASK DATA_SRC VARIANT SCALE RUN EPOCHS SEQ_LEN LR GLOBAL_BATCH MICRO_BATCH
#              BASE_SRC WANDB_GROUP MAX_STEPS NGPU SEED ACT_CKPT SHARD_DEGREE EXTRA (verbatim extra
#              train_ctc_suite.py args, e.g. the --ffn-moe-* / --st-* knobs)
# Placement (env, applied ONLY when this file is executed directly -- it then self-submits):
#              PARTITION (default berkeleynlp) QOS (optional) NODE (optional -w) TIME (default
#              06:00:00) ACCOUNT (optional; jsteinhardt needs ACCOUNT=site QOS=preemptive_high)
#
#   # horton (berkeleynlp), pilot:
#   TASK=contradiction DATA_SRC=/scratch/.../contra_ctc_pilot VARIANT=chunked-mix \
#     RUN=ctc-contra-cmix-08b-pilot ./run_ctc_local.sbatch
#   # mooney (jsteinhardt):
#   PARTITION=jsteinhardt QOS=preemptive_high ACCOUNT=site NODE=mooney TASK=... DATA_SRC=... \
#     VARIANT=chunked RUN=... ./run_ctc_local.sbatch
#   # or plain sbatch (placement via sbatch flags, which override the header):
#   sbatch --partition=jsteinhardt --qos=preemptive_high --account=site -w mooney \
#     --export=ALL,TASK=...,DATA_SRC=...,VARIANT=full,RUN=... run_ctc_local.sbatch
#SBATCH --job-name=ctc-suite
#SBATCH --partition=berkeleynlp
#SBATCH --nodes=1
#SBATCH --gres=gpu:H200:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=/data/prasann/joblogs/ctc_suite_%x_%j.log
# NOTE: --output is node-local /data and must pre-exist on the node -- a job landing on a node
# without /data/prasann/joblogs FAILS in ~2s before the script runs (see the goldgrad launcher).

# ---- self-submit when executed outside slurm: env knobs -> sbatch placement flags ----
if [ -z "${SLURM_JOB_ID:-}" ]; then
  export CTC_LAUNCH_TS="$(date -Is)"   # provenance: true submit time, not job-start time
  exec sbatch \
    --partition="${PARTITION:-berkeleynlp}" \
    ${QOS:+--qos="$QOS"} \
    ${ACCOUNT:+--account="$ACCOUNT"} \
    ${NODE:+-w "$NODE"} \
    --time="${TIME:-06:00:00}" \
    ${RUN:+--job-name="$RUN"} \
    --export=ALL "$0"
fi

set -uo pipefail
# Mirror this job's node-local /data log to shared /scratch so it is readable off-node.
MIRROR_DIR=/scratch/users/prasann/ctc_suite_logs
mkdir -p "$MIRROR_DIR" 2>/dev/null || true
_SRC_LOG="/data/prasann/joblogs/ctc_suite_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
_DST_LOG="$MIRROR_DIR/train_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
_mirror_log() { cp -f "$_SRC_LOG" "$_DST_LOG" 2>/dev/null || true; }
( while true; do _mirror_log; sleep 30; done ) &
_MIRROR_PID=$!
trap '_mirror_log; kill $_MIRROR_PID 2>/dev/null' EXIT

REPO="${REPO:-/accounts/projects/berkeleynlp/prasann/projects/OLMo-core}"
NGPU="${NGPU:-8}"
TASK="${TASK:?set TASK=<suite task name> (e.g. contradiction)}"
DATA_SRC="${DATA_SRC:?set DATA_SRC=<qwen3_5 shard dir on /scratch>}"
VARIANT="${VARIANT:?set VARIANT=full|chunked|chunked-mix}"
SCALE="${SCALE:-0.8b}"
# Model family: empty -> train_ctc_suite.py auto-detects from the shard's metadata marker_set
# (qwen3 plain-dense vs qwen3_5 GDN hybrid). Set explicitly (qwen3|qwen3_5) to force it.
MODEL_FAMILY="${MODEL_FAMILY:-}"
SEQ_LEN="${SEQ_LEN:-40960}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-5e-5}"
GLOBAL_BATCH="${GLOBAL_BATCH:-8}"
MICRO_BATCH="${MICRO_BATCH:-1}"
MAX_STEPS="${MAX_STEPS:-0}"
# Steps between mid-run model+optim checkpoints. Empty/0 = none (the train script's default), so
# existing callers are byte-identical. Set it when you want to EVAL BEFORE THE RUN FINISHES -- with
# no interval the only checkpoint is the model-only one written after fit(), so a 3-hour run yields
# nothing evaluable for 3 hours. Costs ~55G per save at 4B: prune step*/ dirs once consumed.
SAVE_INTERVAL="${SAVE_INTERVAL:-0}"
SEED="${SEED:-0}"
# Optional activation-checkpointing / FSDP shard-degree overrides (mirrors run_ctc_lambda.sbatch).
# Empty by default -> trainer default (auto / no override), so existing callers are unaffected.
ACT_CKPT="${ACT_CKPT:-}"          # full | budget | none | auto
SHARD_DEGREE="${SHARD_DEGREE:-}"  # FSDP shard_degree
# Long-context knobs (empty -> off, so existing callers unaffected). CP_DEGREE = Ulysses degree
# (hybrid<=4, dense<=8); PACK=1 bin-packs whole examples (mixed 8k..256k); AC_BUDGET = budget for
# ACT_CKPT=budget; ROPE_YARN = YaRN factor (dense only).
CP_DEGREE="${CP_DEGREE:-}"
PACK="${PACK:-}"
AC_BUDGET="${AC_BUDGET:-}"
ROPE_YARN="${ROPE_YARN:-}"
ROPE_THETA="${ROPE_THETA:-}"   # NTK context-extension theta (e.g. ~2e6 for Qwen3-4B@64k, ~8e6@256k)
# Optional mask-mix curriculum overrides (chunked-mix variant only). Empty -> train_ctc_suite.py
# defaults (0.80 -> 0.0 anneal). Set BOTH equal for a CONSTANT p (e.g. MIX_START_P=0.5 MIX_END_P=0.5
# to match the paper's fixed p=0.5 instead of the anneal-to-pure-chunked default).
MIX_START_P="${MIX_START_P:-}"
MIX_END_P="${MIX_END_P:-}"
RUN="${RUN:-ctc-${TASK}-${VARIANT}-${SCALE}}"
WANDB_GROUP="${WANDB_GROUP:-ctc-suite-${TASK}}"
# Per-scale converted base (model-only distcp). 0.8B default; 4b/9b MUST pass BASE_SRC (audited).
BASE_SRC="${BASE_SRC:-/scratch/users/prasann/cpt_mix_ckpts/q35-08b-base-modelonly}"
export CTC_LAUNCH_TS="${CTC_LAUNCH_TS:-$(date -Is)}"
echo "=== HOST=$(hostname) TASK=$TASK VARIANT=$VARIANT SCALE=$SCALE SEQ_LEN=$SEQ_LEN RUN=$RUN \
EPOCHS=$EPOCHS GLOBAL_BATCH=$GLOBAL_BATCH START=$(date '+%F %T') ==="

ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$ENV" ] || ENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$REPO/src"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONWARNINGS=ignore
export PYTHONUNBUFFERED=1
# FLA/Triton JIT cache MUST be node-local. It defaults to $HOME/.triton, but $HOME is /accounts
# NFS here; Triton guards each kernel compile with a FileLock, and that lock DEADLOCKS over NFS.
# Two cold-compile jobs on a node (e.g. the synth+baseline arms launched together) then wedge at
# the first forward/backward -- 167 threads stuck in futex_wait, 0% GPU, looks like a dead trainer
# but is really the compile lock. Same class as the flashinfer HOME->NFS wedge, and run_ctc_lambda
# .sbatch already fixes it the same way. A node-local (ext4) FileLock serializes compiles correctly.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/data/prasann/triton_cache}"
mkdir -p "$TRITON_CACHE_DIR" 2>/dev/null || true

# wandb: olmo-core hard-requires the WANDB_API_KEY *env var* and dies with
# `OLMoEnvironmentError: missing env var 'WANDB_API_KEY'` if it is unset -- but it does so only
# AFTER the base checkpoint load and mesh build (~10 min at 128k), and only rank 0 raises, so the
# other ranks spin at 100% GPU in the NCCL collective and SLURM still reports RUNNING. That looks
# exactly like a healthy long-loading job while it burns the whole node. `wandb login` writes the
# key to ~/.netrc but NOT to the env, so a launch from a shell that never exported it hits this.
# Backfill from ~/.netrc (or set NOWANDB=1 to skip wandb entirely).
if [ -z "${WANDB_API_KEY:-}" ] && [ -z "${NOWANDB:-}" ] && [ -r "$HOME/.netrc" ]; then
  WANDB_API_KEY=$(awk '/machine[[:space:]]+api\.wandb\.ai/{f=1} f&&/password/{print $2; exit}' "$HOME/.netrc")
  [ -n "$WANDB_API_KEY" ] && export WANDB_API_KEY && echo "WANDB_API_KEY: backfilled from ~/.netrc"
fi
if [ -z "${WANDB_API_KEY:-}" ] && [ -z "${NOWANDB:-}" ]; then
  echo "FATAL: WANDB_API_KEY unset and not in ~/.netrc -> olmo-core would crash ~10min in with"
  echo "       all ranks hung. Export it, or set NOWANDB=1 to train without wandb."; exit 4
fi

# GPU selection: RESPECT the SLURM allocation when present. On a shared node SLURM's cgroup gives
# us specific devices (CUDA_VISIBLE_DEVICES / SLURM_JOB_GPUS); the old "pick idle GPUs by nvidia-smi
# memory" heuristic can grab devices OUTSIDE our cgroup -> "CUDA driver initialization failed" on the
# extra rank (observed on lorax, which does not index-isolate like thidwick). The clean-pick is kept
# ONLY as a fallback for a bare (non-SLURM-gres) node where nothing set the visibility for us.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] || [ -n "${SLURM_JOB_GPUS:-}" ] || [ -n "${SLURM_STEP_GPUS:-}" ]; then
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<slurm-managed>} (respecting SLURM allocation)"
else
  mapfile -t CLEAN < <(nvidia-smi --query-gpu=memory.used,uuid --format=csv,noheader,nounits | awk -F', ' '$1+0 < 2000 {print $2}')
  export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${CLEAN[*]:0:$NGPU}")
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (clean-pick fallback, no SLURM gres)"
fi

# ---- base: read the shared distcp directly from /scratch; guard the .metadata marker ----
ROOT=/data/prasann/ctc_suite
BASE_CKPT="$BASE_SRC/model_and_optim"
if [ ! -f "$BASE_CKPT/.metadata" ]; then
  echo "FATAL: base checkpoint $BASE_CKPT missing .metadata -> would silently train from scratch"; exit 3
fi

# ---- data + workdir + save folder: node-local /data ----
DATA_LOCAL=$ROOT/data/$(basename "$DATA_SRC")
mkdir -p "$DATA_LOCAL"
# MIRROR, don't "skip if present": cp -un left a stale qwen3_5-tokenized xabsence_train on mooney
# shadowing the olmo3 rebuild of the same basename (2026-08-17, job 3460051 -- caught only by the
# trainer's marker-set guard). rsync --delete is idempotent and transfers nothing when the dir
# already matches, same reasoning as the rung-staging fix in eval_smallscale_ladder.sbatch.
# flock: two jobs staging the same shard concurrently (e.g. paired smokes) race on --delete.
flock "$DATA_LOCAL.lock" rsync -a --delete "$DATA_SRC"/ "$DATA_LOCAL"/ || { echo "FATAL: could not stage $DATA_SRC"; exit 4; }
WORK_LOCAL=$ROOT/cache-$SLURM_JOB_ID
mkdir -p "$WORK_LOCAL"
SAVE_LOCAL=$ROOT/ckpts/$RUN
mkdir -p "$SAVE_LOCAL"
echo "BASE_CKPT=$BASE_CKPT DATA_LOCAL=$DATA_LOCAL SAVE=$SAVE_LOCAL"

cd "$REPO"
MAXARG=""; [ "$MAX_STEPS" -gt 0 ] && MAXARG="--max-steps $MAX_STEPS"
SAVEINTARG=""; [ "$SAVE_INTERVAL" -gt 0 ] && SAVEINTARG="--save-interval $SAVE_INTERVAL"
# NO_FINAL_CKPT=1 skips the end-of-fit FULL (model+optim) save; the model-only export still runs.
# Added after the 2026-08-18 cubbins failure: a completed 1250-step run died writing 82G to a full
# /data and lost its weights entirely -- the model-only export (13G) never got its turn.
NOFINALARG=""; [ -n "${NO_FINAL_CKPT:-}" ] && NOFINALARG="--no-final-checkpoint"
MASTER_PORT=$((29000 + RANDOM % 1000))
# Optional: disable wandb (e.g. no WANDB_API_KEY in the env). Set NOWANDB=1.
# Backward-compatible: empty by default, so existing online-wandb runs are unchanged.
WANDBARG=""; [ -n "${NOWANDB:-}" ] && WANDBARG="--no-wandb"
# Compile: OFF by default (backward-compat; the mask-mix coin isn't compilable anyway). Set
# COMPILE=1 to enable torch.compile -- REQUIRED for --activation-checkpointing budget (which
# olmo-core refuses without compile), and the proven CP recipe (compile + budget AC).
COMPILEARG="--no-compile"; [ -n "${COMPILE:-}" ] && COMPILEARG=""

torchrun --nproc_per_node="$NGPU" --master_port="$MASTER_PORT" \
  src/scripts/train/memexpress/ctc_suite/train_ctc_suite.py \
  --task "$TASK" --variant "$VARIANT" --model-scale "$SCALE" \
  ${MODEL_FAMILY:+--model-family "$MODEL_FAMILY"} \
  --run-name "$RUN" --wandb-group "$WANDB_GROUP" --base-checkpoint "$BASE_CKPT" \
  --data "$DATA_LOCAL" --work-dir "$WORK_LOCAL" --save-folder "$SAVE_LOCAL" \
  --seq-len "$SEQ_LEN" --epochs "$EPOCHS" --lr "$LR" \
  --global-batch "$GLOBAL_BATCH" --micro-batch-instances "$MICRO_BATCH" --seed "$SEED" \
  ${ACT_CKPT:+--activation-checkpointing "$ACT_CKPT"} \
  ${SHARD_DEGREE:+--shard-degree "$SHARD_DEGREE"} \
  ${CP_DEGREE:+--cp-degree "$CP_DEGREE"} \
  ${PACK:+--pack} \
  ${AC_BUDGET:+--ac-budget "$AC_BUDGET"} \
  ${ROPE_YARN:+--rope-yarn-factor "$ROPE_YARN"} \
  ${ROPE_THETA:+--rope-theta "$ROPE_THETA"} \
  ${MIX_START_P:+--mix-start-p "$MIX_START_P"} \
  ${MIX_END_P:+--mix-end-p "$MIX_END_P"} \
  --save-checkpoint $COMPILEARG $WANDBARG \
  $MAXARG $SAVEINTARG $NOFINALARG ${EXTRA:-}
echo "=== DONE $(date '+%F %T')  rc=$? ==="
