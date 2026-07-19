#!/bin/bash
# Stage 1 of the multi-hop gold-routing experiment (records/multihop-gold-routing-experiment.md):
# measure the n=50 floor / ceiling / base-graph routability. ONE ARM PER JOB (pass MODES=<arm>), so the
# three arms run CONCURRENTLY on 2 GPUs each instead of sequentially on 8. Same base + same data + same
# recipe; only --cross-doc-mode differs.
#
# Why 2 GPUs, not 8: global_batch is a fixed 8 instances regardless of world size, so on 8 GPUs each
# rank holds exactly ONE 3136-token sequence -- an H200 at ~12% MFU with nothing to accumulate. On 2
# GPUs each rank holds 4, so --micro-batch-instances 4 does the rank's whole share in one forward.
# Gradient accumulation is exact, so this is mathematically IDENTICAL (same global batch, same 928
# steps, same result as the n=100 recipe) -- purely a throughput/occupancy win. 3 arms x 2 GPUs = 6
# GPUs, inside the per-user cap, and wall-clock collapses to roughly one arm.
#
#   chunked  -> docs fully isolated; all cross-doc work at the trailing FREE tokens. The FLOOR.
#   standard -> plain causal. The CEILING.
#   random   -> random_doc p=0.25, PER-EXAMPLE. The camouflage base graph the hop ladder is built on.
#
# TWO GATES this stage exists to check, before any hop-ladder code is written:
#   1. headroom:  standard - chunked >= 0.2, else n=50 is underpowered -> fall back to n=100.
#   2. routability: `random` must land meaningfully ABOVE the chunked floor. At p=0.25 the doc graph
#      has out-degree 6 and 31% of pairs unreachable at ANY hop -- close to the regime where hier-K10
#      never learned to route and pinned to the chunked floor (f1 0.505 vs 0.441), making per-pair
#      connectivity IRRELEVANT. If `random` pins to the floor, no hop ladder can resolve anything and
#      we fall back to p=0.5.
#
# ⚠ --random-doc-per-example is deliberate and differs from every prior random_doc run (which used ONE
# graph shared across all layers AND all examples). A graph that fixed is memorizable, and later
# stages delete the gold edge -- against a memorized graph a MISSING edge would announce which doc is
# gold. Per-example resampling is what camouflages that. Keep it on so Stage 1 measures the same base
# graph the hop ladder will actually use.
#
# ⚠ Mask mixing is ON (project default) and anneals p_standard 0.8 -> 0.0. VERIFY IT REACHES 0 in the
# log ("mask_mix" lines): under the old NGPU bug it floored at 0.8*(1-1/NGPU), which would keep plain
# causal (and thus a direct gold edge) on ~40% of forwards FOREVER and silently void the experiment.
#
# NFS-safe, per the local-train-nfs-log-hang memory: --output on node-local /data, direct PATH (no
# conda activate), PYTHONWARNINGS=ignore, data+workdir staged node-local, base read IN PLACE (distcp =
# reads; staging via cp drops the hidden .metadata -> silent train-from-scratch).
#
#   for m in chunked standard random; do
#     sbatch --export=ALL,MODES=$m --job-name=q06b-hopgold-$m run_q06b_hopgold_stage1.sbatch
#   done
#   sbatch --export=ALL,MAX_STEPS=3,MODES=chunked run_q06b_hopgold_stage1.sbatch   # smoke
#SBATCH --job-name=q06b-hopgold-s1
#SBATCH --account=site
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --nodes=1
#SBATCH --exclude=mcfuzz,sneetches
#SBATCH --gres=gpu:H200:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=08:00:00
# --output must be node-local /data AND must pre-exist on the node: a job landing on a node without
# /data/prasann/joblogs FAILS in ~2s before this script runs. Restrict via --exclude, NOT --nodelist
# (-w with two names forces BOTH nodes = 16 GPUs -> pends forever on the 8-GPU/user QOS cap).
#SBATCH --output=/data/prasann/joblogs/q06b_hopgold_%x_%j.log

set -uo pipefail
MIRROR_DIR=/scratch/users/prasann/attn_explore_logs
mkdir -p "$MIRROR_DIR" 2>/dev/null || true
_SRC_LOG="/data/prasann/joblogs/q06b_hopgold_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
_DST_LOG="$MIRROR_DIR/hopgold_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log"
_mirror_log() { cp -f "$_SRC_LOG" "$_DST_LOG" 2>/dev/null || true; }
( while true; do _mirror_log; sleep 30; done ) & _MIRROR_PID=$!
trap '_mirror_log; kill $_MIRROR_PID 2>/dev/null' EXIT

REPO="${REPO:-/accounts/projects/berkeleynlp/prasann/projects/OLMo-core}"
NGPU="${NGPU:-2}"
# Instances per device per forward. Must be <= grad_accum/NGPU (the rank's whole share); the launcher
# hard-fails otherwise rather than silently changing the effective global batch.
MICRO_BATCH="${MICRO_BATCH:-4}"
MAX_STEPS="${MAX_STEPS:-0}"
# COMPILE=0 (default) => --no-compile. Two reasons, and the second is the important one:
#   1. Speed: compile costs ~3min of startup. The dry-run compiles once, then the first REAL batch
#      recompiles for its shapes -- observed as a >7min stall after "Starting epoch 1" with 3 jobs
#      contending. Over only ~928 short steps, compile roughly breaks even at best.
#   2. COMPARABILITY (the real reason): Stage 2+'s gold-aware mask CANNOT be compiled -- it needs a
#      per-forward Python fingerprint/RNG hook that torch.compile cannot capture (every goldgrad run
#      is --no-compile for exactly this). If Stage 1 ran compiled, its floor/ceiling would come from a
#      different code path than the hop arms they anchor. Keep every arm eager.
COMPILE="${COMPILE:-0}"
# 0 = no dataloader worker processes. The data is a memory-mapped .npy already staged node-local, so
# workers buy nothing; with several torchrun jobs per node they only add processes that can wedge.
NUM_WORKERS="${NUM_WORKERS:-0}"
# max_example_len=3077 for contra_n50_v2_7k (metadata.json); 3136 = 64*49 fits every example with no
# padding waste. Raising this only burns compute; lowering it DROPS examples.
SEQ_LEN="${SEQ_LEN:-3136}"
EPOCHS="${EPOCHS:-1}"          # 7,421 zero-reuse examples -> repetition is unnecessary
MODES="${MODES:-chunked standard random}"
# Suffix for run names, so a re-run against different DATA_SRC does not collide with (or silently
# resume into) an existing checkpoint dir. Empty = the original recombined-data runs.
TAG="${TAG:-}"
DOC_KEEP_PROB="${DOC_KEEP_PROB:-0.25}"
# summary_attention knobs. SUMMARY_EVERY_K is the CELL SIZE and **must equal the shard's**
# --summary-every-k: the mask identifies a span purely as (chunk_id % (k+1)) == k, so a mismatch
# silently rebinds every chunk role instead of erroring. BANDWIDTH must be <= the SHORTEST span
# (42 tokens, measured) or cell 0 exposes fewer keys than the rest and the dose axis is not uniform.
SUMMARY_EVERY_K="${SUMMARY_EVERY_K:-10}"
BANDWIDTH="${BANDWIDTH:-0}"
RELAY="${RELAY:-1}"
LR="${LR:-5e-5}"
# -trainedmark = the base repaired with TRAINED donor marker rows. NOT -fixmark: that repair fixed the
# marker COSINE but left the marker NORM at ~1/3.6 of a real token, which RMSNorm amplifies into
# full-strength noise and flatlines n=100 chunked training at CE 0.79 for EVERY mask incl. plain
# causal. Any base repaired before 2026-07-14 is affected. See records/n100-chunked-marker-position-bug.md.
BASE_SRC="${BASE_SRC:-/scratch/users/prasann/cpt_mix_ckpts/q06b-dense-cpt-modelonly-trainedmark}"
DATA_SRC="${DATA_SRC:-/scratch/users/prasann/longctx_sft_qwen/contra_n50_v2_7k}"
SAVE_ROOT="${SAVE_ROOT:-/scratch/users/prasann/olmo_ckpts}"
echo "=== HOST=$(hostname) MODES='$MODES' SEQ_LEN=$SEQ_LEN EPOCHS=$EPOCHS P=$DOC_KEEP_PROB MAX_STEPS=$MAX_STEPS START=$(date '+%F %T') ==="

# ENV_ROOT=scratch forces the /scratch env, pinned to the KNOWN-GOOD flash-attn 2.8.2 (2.8.3 has a
# varlen-backward bug that SIGSEGVs mid-run -- see the flash-attn-sigsegv-fix memory).
ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ "${ENV_ROOT:-data}" = "scratch" ] && ENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
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

mapfile -t CLEAN < <(nvidia-smi --query-gpu=memory.used,uuid --format=csv,noheader,nounits | awk -F', ' '$1+0 < 2000 {print $2}')
export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${CLEAN[*]:0:$NGPU}")
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

BASE_CKPT="$BASE_SRC/model_and_optim"
if [ ! -f "$BASE_CKPT/.metadata" ]; then
  echo "FATAL: base checkpoint $BASE_CKPT missing .metadata -> would silently train from scratch"; exit 3
fi

ROOT=/data/prasann/hopgold
DATA_LOCAL=$ROOT/data/$(basename "$DATA_SRC")
mkdir -p "$DATA_LOCAL"
cp -un "$DATA_SRC"/* "$DATA_LOCAL"/ 2>/dev/null || cp -n "$DATA_SRC"/* "$DATA_LOCAL"/ 2>/dev/null || true
[ -f "$DATA_LOCAL/metadata.json" ] || { echo "FATAL: data missing in $DATA_LOCAL"; exit 4; }

cd "$REPO"
MAXARG=""; [ "$MAX_STEPS" -gt 0 ] && MAXARG="--max-steps $MAX_STEPS"
COMPILEARG=""; [ "$COMPILE" -eq 0 ] && COMPILEARG="--no-compile"

_IDX=0
for MODE in $MODES; do
  EXTRA=""
  case "$MODE" in
    chunked)  XDOC=chunked;  RUN="q06b-hopgold-n50${TAG}-chunked" ;;
    standard) XDOC=standard; RUN="q06b-hopgold-n50${TAG}-full" ;;
    random)   XDOC=random_doc; RUN="q06b-hopgold-n50${TAG}-rand${DOC_KEEP_PROB}"
              EXTRA="--doc-keep-prob $DOC_KEEP_PROB --random-doc-per-example" ;;
    # summary-attention bandwidth ladder. ONE shard (built --summary-every-k K) serves EVERY rung:
    # bandwidth/relay are mask knobs, so the data is bit-identical across rungs and only visibility
    # changes. b=0 reproduces the cell-blocks floor exactly. RELAY=0 = the placebo (span reads nothing
    # -> provably carries zero document content); run it at the TOP rung, where it must land on b=0.
    summary)  XDOC=summary_attention
              RUN="q06b-hopgold-n50${TAG}-sum${SUMMARY_EVERY_K}b${BANDWIDTH}"
              EXTRA="--summary-every-k $SUMMARY_EVERY_K --summary-bandwidth $BANDWIDTH"
              if [ "${RELAY:-1}" -eq 0 ]; then
                EXTRA="$EXTRA --no-summary-relay"; RUN="${RUN}-placebo"
              fi ;;
    *) echo "unknown MODE=$MODE"; exit 5 ;;
  esac
  WORK_LOCAL=$ROOT/cache-$SLURM_JOB_ID-$MODE; mkdir -p "$WORK_LOCAL"
  SAVE_FOLDER="$SAVE_ROOT/$RUN"
  # ⚠ Trainer SILENTLY auto-resumes from an existing --save-folder (starts at step N, not 1), which
  # makes step counts and any wall-clock/speedup number garbage and can inherit a poisoned base.
  if [ -d "$SAVE_FOLDER" ] && [ "$MAX_STEPS" -eq 0 ]; then
    echo "FATAL: $SAVE_FOLDER exists -> Trainer would silently RESUME, not restart. Move/delete it."; exit 6
  fi
  # Derive the port from the JOB ID, not $RANDOM: co-tenant torchrun jobs using 29000+RANDOM can
  # collide, rendezvous against the WRONG master, and hang until the 30-min DistStoreError timeout.
  MASTER_PORT=$((29000 + (SLURM_JOB_ID + _IDX) % 2000))
  _IDX=$((_IDX + 1))
  echo "=== [$MODE] RUN=$RUN SAVE=$SAVE_FOLDER $(date '+%F %T') ==="
  torchrun --nproc_per_node="$NGPU" --master_port="$MASTER_PORT" \
    src/scripts/train/memexpress/attn_explore/Qwen3-0.6B-docchunk-mask-mix-contradiction-SFT-local.py \
    --run-name "$RUN" --base-checkpoint "$BASE_CKPT" \
    --data-dir "$DATA_LOCAL" --work-dir "$WORK_LOCAL" --save-folder "$SAVE_FOLDER" \
    --cross-doc-mode "$XDOC" $EXTRA \
    --seq-len "$SEQ_LEN" --epochs "$EPOCHS" --lr "$LR" --grad-accum 8 \
    --micro-batch-instances "$MICRO_BATCH" --num-workers "$NUM_WORKERS" $COMPILEARG \
    --mix-mode curriculum --mix-start-p 0.8 --mix-end-p 0.0 \
    --save-checkpoint --wandb-group "hopgold-n50-stage1" $MAXARG $WANDB_FLAG
  echo "=== [$MODE] rc=$? done $(date '+%F %T') ==="
done
echo "=== ALL DONE $(date '+%F %T') ==="
