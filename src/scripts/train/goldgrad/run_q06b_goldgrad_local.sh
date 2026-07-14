#!/bin/bash
# Gold-grad (O(1)-backward) arms run DIRECTLY on the GPUs of the current interactive session
# (no sbatch queue). Sequential arms, one torchrun each.
#
# Why this is comparable to the 8-GPU sbatch runs: the launcher's global batch is
# `--grad-accum * seq_len` TOKENS and is independent of world size (each rank just does more
# microbatches). So 4 GPUs vs 8 gives the SAME global batch and the SAME step count -- only
# wall-clock differs. Do not "compensate" by changing --grad-accum.
#
# Base MUST be the marker-repaired (-fixmark) checkpoint: on a stock Qwen3 base the
# <|box_start|>/<|box_end|> rows are bit-identical (never trained), which at 100 docs yields low train
# CE but ~0 held-out f1. See records/document-chunked-marker-embeddings.md.
#
#   bash src/scripts/train/goldgrad/run_q06b_goldgrad_local.sh n100
#   bash src/scripts/train/goldgrad/run_q06b_goldgrad_local.sh n20
set -uo pipefail

RUNG="${1:-n100}"
REPO="${REPO:-/accounts/projects/berkeleynlp/prasann/projects/OLMo-core}"
NGPU="${NGPU:-$(nvidia-smi -L | wc -l)}"
TAG="${TAG:-famark}"   # famark = full attention + marker-repaired base
BASE="${BASE:-/scratch/users/prasann/cpt_mix_ckpts/q06b-dense-cpt-modelonly-fixmark/model_and_optim}"
SAVE_ROOT="${SAVE_ROOT:-/data/prasann/olmo_ckpts}"
LOGDIR="${LOGDIR:-/data/prasann/goldgrad_local}"
EVAL_SIZE="${EVAL_SIZE:-488}"   # whole contra eval file; never the 100 default (SE +/-0.046 at 100)
mkdir -p "$LOGDIR" "$SAVE_ROOT"

case "$RUNG" in
  n20)  DATA=/scratch/users/prasann/longctx_sft_qwen/contradiction_n20_docdense_nocot_gold;  SEQ=2048; EPOCHS=3  ;;
  # n100 uses the 2000-example shard (same example count + epochs as n20, so the rungs differ ONLY in
  # document count). Its gold sidecar was recovered from the shard's own answer tokens -- the gold docs
  # ARE the answer -- via build_gold_sidecar_from_shard.py (validated 2000/2000 against n20's reference
  # sidecar). The earlier 500-example shard (shrunk from n190) is retired: 4x less data + 10 epochs made
  # it memorize, and it wasn't comparable to n20.
  n100) DATA=/scratch/users/prasann/longctx_sft_qwen/contradiction_n100_docdense_nocot_gold2k; SEQ=6144; EPOCHS=3 ;;
  *) echo "usage: $0 {n20|n100}"; exit 2 ;;
esac
# --plain-attention at n100: FULL attention is what we want to study here (no chunked mask), and plain
# causal is MATH-IDENTICAL to our mask anyway (random_doc @ doc_keep_prob=1.0 is provably plain causal:
# 0 causal-allowed positions blocked) while being faster.
#
# CORRECTION (do not reintroduce): an earlier version of this comment claimed the doc-chunked DENSE-MASK
# path "SIGSEGVs deterministically at step ~330 at seq 6144". THAT WAS FALSE -- no dense-mask bug was
# ever independently reproduced. A follow-up comment then blamed faulthandler's 300s watchdog; that was
# ALSO false (it is exit=False, and the `full` arm fired it 4x and still finished 750/750 steps).
# The intermittent `exitcode: -11` on these local runs remains UNEXPLAINED. It is intermittent, not
# deterministic. Prime suspect is the known flash-attn 2.8.3 varlen-backward SIGSEGV (pin 2.8.2).
# Do not add a sixth theory here without a controlled test.
PLAIN=""; [ "$RUNG" = "n100" ] && PLAIN="--plain-attention"
[ -f "$BASE/.metadata" ] || { echo "FATAL: base $BASE missing .metadata (would train from scratch)"; exit 3; }
[ -f "$DATA/gold_fingerprints.json" ] || { echo "FATAL: gold sidecar missing in $DATA"; exit 4; }

ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$ENV" ] || ENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$REPO/src"
export TOKENIZERS_PARALLELISM=false PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [ -z "${WANDB_API_KEY:-}" ]; then
  WANDB_API_KEY=$(awk '/machine api.wandb.ai/{f=1} f&&/password/{print $2; exit}' "$HOME/.netrc" 2>/dev/null); export WANDB_API_KEY
fi
# WANDB ON by default (set WANDB=0 to disable). It was briefly turned off while wandb's sync thread was
# (wrongly) suspected of causing the 343s "SIGSEGV" -- the real cause was faulthandler's 300s watchdog,
# and the one run that "proved" wandb guilty was simply the only short one. wandb is exonerated.
WANDB_FLAG=""; { [ "${WANDB:-1}" = "0" ] || [ -z "${WANDB_API_KEY:-}" ]; } && WANDB_FLAG="--no-wandb"

# arm := "<grad-mode>:<n_gold>:<n_random>:<suffix>"
ARMS="${ARMS:-full:0:0:full gold_plus_random:0:2:gpr2 random_only:0:2:rand2 gold_subsample:1:15:gsub1_15}"

echo "=== HOST=$(hostname) NGPU=$NGPU RUNG=$RUNG SEQ=$SEQ EPOCHS=$EPOCHS BASE=$(basename $(dirname $BASE)) START=$(date '+%F %T') ==="
for arm in $ARMS; do
  IFS=: read -r MODE NGOLD NRAND SUF <<< "$arm"
  RUN="q06b-goldgrad-${TAG}-${RUNG}-${SUF}"
  LOG="$LOGDIR/${RUN}.log"

  # The Trainer AUTO-RESUMES from any stepN checkpoint already in --save-folder, silently: it logs
  # "Will resume training from step 250" and then runs only the REMAINING steps. That is a trap --
  # a relaunched arm looks like a fresh run but finishes in a fraction of the wall-clock (which is how
  # a 134s-vs-340s ratio once got misread as a 2.5x backward speedup), and if the stale checkpoint came
  # from a different base or --grad-mode you silently inherit it. Force an explicit choice.
  if compgen -G "$SAVE_ROOT/$RUN/step*" > /dev/null; then
    if [ "${RESUME:-0}" = "1" ]; then
      echo "=== [$MODE] RESUME=1: continuing existing $RUN (steps: $(ls -1d $SAVE_ROOT/$RUN/step* | sed 's/.*step//' | sort -n | tr '\n' ' ')) ==="
    elif [ "${FRESH:-0}" = "1" ]; then
      echo "=== [$MODE] FRESH=1: wiping existing checkpoints for $RUN ==="; rm -rf "$SAVE_ROOT/$RUN"
    else
      echo "=== [$MODE] SKIP: $SAVE_ROOT/$RUN already has checkpoints. Would SILENTLY RESUME."
      echo "===          Re-run with FRESH=1 (retrain from base) or RESUME=1 (intentionally continue). ==="
      continue
    fi
  fi

  echo "=== [$MODE] RUN=$RUN -> $LOG  $(date '+%F %T') ==="
  torchrun --nproc_per_node="$NGPU" --master_port=$((29100 + RANDOM % 800)) \
    "$REPO/src/scripts/train/goldgrad/Qwen3-0.6B-goldgrad-contradiction-n20-SFT-local.py" \
    --run-name "$RUN" --base-checkpoint "$BASE" \
    --data-dir "$DATA" --work-dir "/data/prasann/goldgrad/cache-$RUN" \
    --save-folder "$SAVE_ROOT/$RUN" \
    --grad-mode "$MODE" --n-random "$NRAND" --n-gold "$NGOLD" \
    --cross-doc-mode random_doc --doc-keep-prob 1.0 $PLAIN \
    --seq-len "$SEQ" --epochs "$EPOCHS" --grad-accum 8 --num-workers 0 \
    --save-checkpoint --wandb-group "goldgrad-${RUNG}-${TAG}" $WANDB_FLAG > "$LOG" 2>&1
  rc=$?
  ce=$(grep -oE "train/CE loss=[0-9.]+" "$LOG" | tail -1)
  echo "=== [$MODE] train rc=$rc  last $ce  $(date '+%F %T') ==="

  # ---- eval THIS arm immediately, on the same GPUs, before moving to the next ----
  # (a) results land incrementally instead of after every arm trains;
  # (b) the checkpoint is node-local, so evaluating here avoids ever moving it.
  if [ "${EVAL:-1}" = "1" ] && [ -f "$SAVE_ROOT/$RUN/model_and_optim/.metadata" ]; then
    echo "=== [$MODE] eval start $(date '+%F %T') ==="
    EVAL_SIZE="$EVAL_SIZE" RUNS="$RUN" TAG="$TAG" CKPT_ROOT="$SAVE_ROOT" NGPU="$NGPU" \
      bash "$REPO/src/scripts/train/goldgrad/eval_q06b_goldgrad_local.sh" "$RUNG" 2>&1 | grep -E "f1=|SKIP|rc=|Traceback"
    echo "=== [$MODE] eval done $(date '+%F %T') ==="
  elif [ "${EVAL:-1}" = "1" ]; then
    echo "=== [$MODE] SKIP eval: no checkpoint (train crashed) ==="
  fi
done
echo "=== ALL DONE $(date '+%F %T') ==="
