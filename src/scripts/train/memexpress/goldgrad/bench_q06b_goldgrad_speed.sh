#!/bin/bash
# Controlled wall-clock benchmark for the gold-grad O(1)-backward claim.
#
# WHY THIS EXISTS: you cannot read the speedup off the training-job durations. The Trainer silently
# AUTO-RESUMES from any checkpoint already in --save-folder, so a relaunched arm executes only the
# remaining steps and "finishes" in a fraction of the wall-clock -- which looks exactly like a speedup.
# (That is precisely how a 134s-vs-340s ratio got misread as 2.5x when the arm had merely run 250 of
# 750 steps.) So: FRESH save folder per arm, fixed --max-steps, no checkpoint save, same GPUs, same
# data, same seq len. The ONLY thing that varies is --grad-mode.
#
#   bash src/scripts/train/memexpress/goldgrad/bench_q06b_goldgrad_speed.sh n100
set -uo pipefail

RUNG="${1:-n100}"
STEPS="${STEPS:-40}"          # enough to average out warmup; first steps are dropped below
REPO="${REPO:-/accounts/projects/berkeleynlp/prasann/projects/OLMo-core}"
NGPU="${NGPU:-$(nvidia-smi -L | wc -l)}"
BASE="${BASE:-/scratch/users/prasann/cpt_mix_ckpts/q06b-dense-cpt-modelonly-fixmark/model_and_optim}"
OUT="${OUT:-$REPO/debug/goldgrad_speed}"
mkdir -p "$OUT"

case "$RUNG" in
  n20)  DATA=/scratch/users/prasann/longctx_sft_qwen/contradiction_n20_docdense_nocot_gold;    SEQ=2048 ;;
  n100) DATA=/scratch/users/prasann/longctx_sft_qwen/contradiction_n100_docdense_nocot_gold2k; SEQ=6144 ;;
  *) echo "usage: $0 {n20|n100}"; exit 2 ;;
esac
PLAIN=""; [ "$RUNG" = "n100" ] && PLAIN="--plain-attention"

ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
[ -d "$ENV" ] || ENV=/scratch/users/prasann/conda/envs/corpus-reasoning-olmo
export PATH="$ENV/bin:$PATH"
export PYTHONPATH="$REPO/src"
export TOKENIZERS_PARALLELISM=false PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# arm := "<grad-mode>:<n_gold>:<n_random>:<suffix>"
ARMS="${ARMS:-full:0:0:full gold_plus_random:0:2:gpr2 gold_subsample:1:15:gsub1_15}"

echo "=== SPEED BENCH  rung=$RUNG seq=$SEQ steps=$STEPS ngpu=$NGPU  $(date '+%F %T') ==="
for arm in $ARMS; do
  IFS=: read -r MODE NGOLD NRAND SUF <<< "$arm"
  RUN="bench-${RUNG}-${SUF}"
  SAVE="/data/prasann/olmo_ckpts/_bench/$RUN"
  rm -rf "$SAVE"                      # MUST be fresh: a stale ckpt would auto-resume and fake the timing
  LOG="$OUT/${RUN}.log"

  torchrun --nproc_per_node="$NGPU" --master_port=$((29900 + RANDOM % 90)) \
    "$REPO/src/scripts/train/memexpress/goldgrad/Qwen3-0.6B-goldgrad-contradiction-n20-SFT-local.py" \
    --run-name "$RUN" --base-checkpoint "$BASE" \
    --data-dir "$DATA" --work-dir "/data/prasann/goldgrad/cache-bench-$RUNG" \
    --save-folder "$SAVE" \
    --grad-mode "$MODE" --n-random "$NRAND" --n-gold "$NGOLD" \
    --cross-doc-mode random_doc --doc-keep-prob 1.0 $PLAIN \
    --seq-len "$SEQ" --epochs 1 --grad-accum 8 --num-workers 0 \
    --max-steps "$STEPS" --no-wandb > "$LOG" 2>&1
  rc=$?

  # The trainer emits `throughput/device/TPS` (tokens/s/device) and MFU -- there is NO "step time" metric
  # (an earlier version of this script grepped for one and silently printed 0.0000). Metrics are logged
  # every ~10 steps, so STEPS=40 yields only ~4 samples; drop the 1st (warmup) and average the rest.
  read -r n tps mfu <<< "$(paste -d' ' \
      <(grep -oE "throughput/device/TPS=[0-9.]+" "$LOG" | sed 's/.*=//') \
      <(grep -oE "throughput/device/MFU=[0-9.]+" "$LOG" | sed 's/.*=//') \
      | awk 'NR>1{t+=$1; m+=$2; n++} END{printf "%d %.1f %.1f", n+0, (n?t/n:0), (n?m/n:0)}')"
  det=$(grep -oE "detached=[0-9]+/[0-9]+" "$LOG" | tail -1)
  first=$(grep -oE "step=[0-9]+/[0-9]+" "$LOG" | head -1)
  printf "  %-14s rc=%s  n=%-2s  TPS/dev=%-7s MFU=%-6s %-22s first=%s\n" \
    "$SUF" "$rc" "$n" "$tps" "$mfu" "${det:-detached=0/N}" "${first:-?}"
  rm -rf "$SAVE"
done
echo "=== BENCH DONE $(date '+%F %T') ==="
echo "RESULT (2026-07-13, n100 seq6144, 4xH200): ALL arms 30 TPS/dev @ 76% MFU -> speedup = 1.00x."
echo "Detaching 81% of context tokens buys NOTHING. torch.where(keep, k, k.detach()) removes no compute:"
echo "grad wrt k is still a full-size DENSE tensor (zeros in detached rows) and the backward of W_k @ h"
echo "is still a dense matmul over all positions. The backward is sparse IN VALUE, not IN COMPUTE."
echo "A real O(1) backward must physically gather the kept docs' tokens so the matmuls actually shrink."
echo "Forward stays full by design -> ceiling is ~2x, never more."
