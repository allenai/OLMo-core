#!/bin/bash
# One olmo3_cptdrop training run (records/olmo3-cptdrop-screen-plan.md) on a Berkeley node that holds
# /data/prasann/olmo3_cptdrop/{data,base} (mooney; else pull them with s3_sync.py down first).
# Code is a staged tree at $ROOT/code_<sha> (git archive HEAD | srun ... tar -x), never the NFS repo.
#
#   ARM=D RUN=o3cd-D-480 MAX_STEPS_TOTAL=480 SAVE_INTERVAL=108 CODE_SHA=<sha> \
#     sbatch -w mooney --gres=gpu:8 research/token_routing/olmo3_cptdrop/run_arm.sbatch
#
# Knobs (env):
#   ARM            D | C1 (8k phase) | C2 (32k phase) | IRAND | IFL
#                  follow-ups: IFL_R4K | IFL_RAMP | IFL_X4 | IFL_X8 | IRAND_X4 | IRAND_X8
#   RUN            run name = save folder basename (must be NEW unless RESUME=1)
#   MAX_STEPS_TOTAL  budget in optimizer steps at 524,288 tokens/step (sets --max-tokens)
#   HARD_STOP      --max-steps (arm C phase 1 stops at half its budget, LR still at peak)
#   SAVE_INTERVAL  mid-run full checkpoints (0.9 x a smaller budget = a decay-branch point)
#   BRANCH_FROM    stepN dir to branch from; BRANCH_OPTIM_ONLY=1 for arm C phase 2
#   WARMUP / DECAY_FRAC   WSD warmup fraction (0.017) / decay fraction (0.1)
#   SEED, EXTRA (appended trainer flags), NOWANDB=1, CODE_SHA (staged tree)
#   CTX=64k        64k rows (train64k), 8 rows/step, YaRN x8      KEEP=<f>  drop-arm keep fraction / prob
#   ST_EXTRA       extra --st-* flags (e.g. --st-keep-recent 4096 once that lands)
#   SMOKE=1        no model-only export and NO checkpoints at all (--no-final-checkpoint disables the
#                  checkpointer callback, mid-run SAVE_INTERVAL saves included)
#SBATCH --job-name=o3cd-arm
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --cpus-per-task=64
#SBATCH --mem=640G
#SBATCH --time=12:00:00
#SBATCH --output=/data/prasann/olmo3_cptdrop/logs/%x-%j.out
set -uo pipefail
# ROOT / PY / CODE are overridable so beaker_job.sh can run the same recipe off-cluster.
ROOT="${ROOT:-/data/prasann/olmo3_cptdrop}"
REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
PY="${PY:-$ENV/bin/python}"
CODE="${CODE:-$ROOT/code_${CODE_SHA:?set CODE_SHA to a staged tree under $ROOT (or CODE=)}}"
ARM="${ARM:?ARM=D|C1|C2|IRAND|IFL}"
RUN="${RUN:?RUN=<run name>}"
NGPU="${NGPU:-$(nvidia-smi -L | wc -l)}"
TOK_PER_STEP=524288
STEPS="${MAX_STEPS_TOTAL:?MAX_STEPS_TOTAL=<budget in steps>}"
[ -d "$CODE/src/olmo_core" ] || { echo "FATAL: no staged code at $CODE"; exit 2; }
[ -x "$PY" ] || { echo "FATAL: no interpreter $PY"; exit 2; }

case "$ARM" in
  D)     DATA=train32k SEQ=32768 GB=16 VAR="--variant full" ;;
  C1)    DATA=c8k      SEQ=8192  GB=64 MB=4 VAR="--variant full" ;;  # 4 x 8k = 32k tokens per micro-batch, as the 32k arms
  C2)    DATA=train32k_head SEQ=32768 GB=16 VAR="--variant full" ;;  # rows 0..39,159: never the tail rows c8k was cut from
  IRAND) DATA=train32k SEQ=32768 GB=16 VAR="--variant softtoken --st-block-size 512 --st-gold-blind --st-drop-slots --st-keep-prob 0.2 --st-pad-multiple 1" ;;
  IFL)   DATA=train32k SEQ=32768 GB=16 VAR="--variant softtoken --st-block-size 512 --st-gold-blind --st-drop-slots --st-keep-prob 0.0 --st-keep-token-rule first_last --st-keep-token-k 0.25 --st-pad-multiple 1" ;;
  # follow-up arms (2026-10-22, not in the round-1 plan): position-aware keep rules and token exit.
  # L=4 / L=8 = dropped tokens leave after the first / second full-attention layer (3, 7).
  IFL_R4K)  DATA=train32k SEQ=32768 GB=16 VAR="--variant softtoken --st-block-size 512 --st-gold-blind --st-drop-slots --st-keep-prob 0.0 --st-keep-token-rule first_last --st-keep-token-k 0.25 --st-pad-multiple 1 --st-keep-recent 4096" ;;
  IFL_RAMP) DATA=train32k SEQ=32768 GB=16 VAR="--variant softtoken --st-block-size 512 --st-gold-blind --st-drop-slots --st-keep-prob 0.0 --st-keep-token-rule first_last --st-keep-token-k 0.25 --st-pad-multiple 1 --st-keep-ramp 0.1,0.5" ;;
  IFL_X4)   DATA=train32k SEQ=32768 GB=16 VAR="--variant softtoken --st-block-size 512 --st-gold-blind --st-drop-slots --st-keep-prob 0.0 --st-keep-token-rule first_last --st-keep-token-k 0.25 --st-pad-multiple 1 --tokexit-layer 4" ;;
  IFL_X8)   DATA=train32k SEQ=32768 GB=16 VAR="--variant softtoken --st-block-size 512 --st-gold-blind --st-drop-slots --st-keep-prob 0.0 --st-keep-token-rule first_last --st-keep-token-k 0.25 --st-pad-multiple 1 --tokexit-layer 8" ;;
  IRAND_X4) DATA=train32k SEQ=32768 GB=16 VAR="--variant softtoken --st-block-size 512 --st-gold-blind --st-drop-slots --st-keep-prob 0.2 --st-pad-multiple 1 --tokexit-layer 4" ;;
  IRAND_X8) DATA=train32k SEQ=32768 GB=16 VAR="--variant softtoken --st-block-size 512 --st-gold-blind --st-drop-slots --st-keep-prob 0.2 --st-pad-multiple 1 --tokexit-layer 8" ;;
  *) echo "FATAL: unknown ARM=$ARM"; exit 2 ;;
esac
# KEEP overrides the drop arms' aggressiveness (I-rand: block keep-prob; I-fl: first_last fraction per block)
if [ -n "${KEEP:-}" ]; then
  case "$ARM" in
    IRAND) VAR="${VAR/--st-keep-prob 0.2/--st-keep-prob $KEEP}" ;;
    IFL)   VAR="${VAR/--st-keep-token-k 0.25/--st-keep-token-k $KEEP}" ;;
    *) echo "FATAL: KEEP only applies to IRAND / IFL"; exit 2 ;;
  esac
fi
# CTX=64k: one 64k window per row, 8 rows/step (same 524,288 tokens/step), YaRN x8 (8192 -> 65536, as OLMo stage 3)
YARN=4
if [ "${CTX:-32k}" = "64k" ]; then
  case "$ARM" in D|IRAND|IFL) DATA=train64k; SEQ=65536; GB=8; YARN=8 ;; *) echo "FATAL: CTX=64k not wired for ARM=$ARM"; exit 2 ;; esac
fi
VAR="$VAR ${ST_EXTRA:-}"
SAVE="$ROOT/runs/$RUN"
if [ -d "$SAVE" ] && ls -d "$SAVE"/step* >/dev/null 2>&1 && [ -z "${RESUME:-}" ]; then
  echo "FATAL: $SAVE already has step checkpoints; the trainer would silently resume (set RESUME=1 to mean it)"; exit 3
fi

export PATH="$(dirname "$PY"):$PATH" PYTHONPATH="$CODE/src:$CODE/recipes/train/research_trainer:$CODE/recipes/train:$CODE"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
TMPBASE="${TMPBASE:-$ROOT/tmp}"   # Beaker: node-local /tmp while ROOT is on weka (JIT caches stay off the network fs)
export HF_HOME="$ROOT/hf_cache" TMPDIR="$TMPBASE" TRITON_CACHE_DIR="$TMPBASE/triton" TORCHINDUCTOR_CACHE_DIR="$TMPBASE/inductor"
mkdir -p "$TMPDIR" "$TRITON_CACHE_DIR" "$ROOT/runs" "$ROOT/work"
if [ -z "${NOWANDB:-}" ] && [ -z "${WANDB_API_KEY:-}" ] && [ -r "$HOME/.netrc" ]; then
  WANDB_API_KEY=$(awk '/machine[[:space:]]+api\.wandb\.ai/{f=1} f&&/password/{print $2; exit}' "$HOME/.netrc"); export WANDB_API_KEY
fi
[ -n "${NOWANDB:-}" ] || [ -n "${WANDB_API_KEY:-}" ] || { echo "FATAL: no WANDB_API_KEY (or set NOWANDB=1)"; exit 4; }

ARGS=(--task cpt --model-family olmo3 --model-scale 7b $VAR
  --data "$ROOT/data/$DATA" --seq-len "$SEQ" --global-batch "$GB" --micro-batch-instances "${MB:-1}" --epochs 1
  --base-checkpoint "$ROOT/base/stage2-step47684/model_and_optim" --skip-marker-audit --rope-yarn-factor "$YARN"
  --lr 2.0712352850360292e-4 --weight-decay 0.1 --z-loss 1e-5
  --scheduler wsd --warmup-fraction "${WARMUP:-0.017}" --wsd-decay-fraction "${DECAY_FRAC:-0.1}"
  --max-tokens $(( STEPS * TOK_PER_STEP ))
  --activation-checkpointing full --shard-degree "$NGPU" --attn-backend flash_2 --no-compile
  --save-folder "$SAVE" --work-dir "$ROOT/work/$RUN" --run-name "$RUN"
  --wandb-group "${WANDB_GROUP:-o3cd-q3-7b}" --seed "${SEED:-0}")
if [ -n "${SMOKE:-}" ]; then ARGS+=(--no-final-checkpoint); else ARGS+=(--save-checkpoint); fi
[ -n "${HARD_STOP:-}" ] && ARGS+=(--max-steps "$HARD_STOP")
[ -n "${SAVE_INTERVAL:-}" ] && ARGS+=(--save-interval "$SAVE_INTERVAL")
[ -n "${BRANCH_FROM:-}" ] && ARGS+=(--branch-from "$BRANCH_FROM")
[ -n "${BRANCH_OPTIM_ONLY:-}" ] && ARGS+=(--branch-optim-only)
[ -n "${NOWANDB:-}" ] && ARGS+=(--no-wandb)
[ -n "${RESUME:-}" ] && ARGS+=(--resume)
# shellcheck disable=SC2206
ARGS+=(${EXTRA:-})
echo "=== $(hostname) ARM=$ARM RUN=$RUN NGPU=$NGPU STEPS=$STEPS CODE=$CODE start $(date '+%F %T') ==="
printf ' %q' "${ARGS[@]}"; echo
cd "$ROOT"
CMD=("$PY" -m torch.distributed.run --standalone --nproc_per_node="$NGPU"
  "$CODE/recipes/train/research_trainer/train_ctc_suite.py" "${ARGS[@]}")
if [ -f "$REPO/recipes/env/ops_job.sh" ]; then   # Berkeley: register + guard
  OPS_RUN_ID="$RUN" OPS_EXPERIMENT=olmo3_cptdrop OPS_ARM="$ARM" OPS_CKPT_PATH="$SAVE" \
  OPS_EXPECTED_MINUTES="${EXPECTED_MIN:-120}" OPS_FIRST_OUTPUT=1500 OPS_STALL=1800 \
  bash "$REPO/recipes/env/ops_job.sh" "${CMD[@]}"
else                                              # Beaker: the experiment is registered at launch
  "${CMD[@]}"
fi
