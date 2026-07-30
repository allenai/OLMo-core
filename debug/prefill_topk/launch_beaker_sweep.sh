#!/bin/bash
# Submit the 4B compressive-landmark prefill-top-k sweep as ONE BEAKER JOB PER CONFIG, so the four
# configs run concurrently instead of serially (bs=1 landmark decode over 4 rungs x 500 examples is
# ~3.5 GPU-h per config; serial would be a >14 h job).
#
#   bash debug/prefill_topk/launch_beaker_sweep.sh              # submit
#   DRY=1 bash debug/prefill_topk/launch_beaker_sweep.sh        # build + print only
#
# ⚠ commit AND push first -- gantry clones the repo at HEAD.
set -uo pipefail
REPO="${REPO:-/accounts/projects/berkeleynlp/prasann/projects/OLMo-core}"
RUN="${RUN:-q4b-compressive-5task-32k-nocpt-fixdata}"
CLUSTER="${CLUSTER:-ai2/jupiter}"
NGPU="${NGPU:-4}"
DRY_FLAG=""; [ "${DRY:-0}" = "1" ] && DRY_FLAG="--dry-run"

cd "$REPO"
export PYTHONPATH="$REPO/src"

CONFIGS=(
  "baseline_decode_only|"
  "prefill_topk10pct|--prefill-topk-fraction 0.1 --prefill-nonselected-mass 0"
  "prefill_topk25pct|--prefill-topk-fraction 0.25 --prefill-nonselected-mass 0"
  "prefill_topk50pct|--prefill-topk-fraction 0.5 --prefill-nonselected-mass 0"
)

for entry in "${CONFIGS[@]}"; do
  TAG="${entry%%|*}"
  echo "=== submitting $TAG ==="
  python debug/prefill_topk/launch_beaker_prefill_topk_eval.py "$RUN" "$CLUSTER" \
    --task contradiction --rungs 2k,8k,16k,32k --ngpu "$NGPU" \
    --configs "$entry" $DRY_FLAG
done
echo "=== done: ${#CONFIGS[@]} jobs ==="
