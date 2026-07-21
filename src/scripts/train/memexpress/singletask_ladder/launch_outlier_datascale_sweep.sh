#!/bin/bash
# OUTLIER TRAINING-DATA-SCALE SWEEP -- chunked + sparse-landmark arms.
#
# Replicates the dense sweep `q4b-dense-outlier-ladder32k-ss{2p5,5,10,25,50,100}` (evaluated
# 2026-07-09, results-hub `attention_type=full`), which showed f1 rising monotonically with training
# data at every rung (3k: 0.35 -> 0.89; 32k: 0.07 -> 0.27 from 207 to 8302 examples). Same task, same
# ladder shards, same 6 subsample fractions, same 1 epoch -- only the attention changes:
#
#   docchunk        DocumentChunkedAttention (cross_doc_mode="chunked"), box-marker shards
#                   -> Qwen3-4B-docchunk-singletask-ladder-10k-SFT.py
#   sparselandmark  AttentionType.sparse_landmark: full attention within a chunk, past chunks
#                   visible ONLY through their single landmark token (no expansion)
#                   -> Qwen3-4B-singletask-ladder-32k-10k-3variant-SFT.py
#
# Both launchers read the fraction from STL_SUBSAMPLE and re-export it as a Beaker env var, because
# the Beaker job REBUILDS the config on the node -- a fraction resolved only on the launch host would
# silently fall back to the module default and every sweep point would train on identical data.
# The docchunk launcher additionally takes STL_EPOCHS=1 here; its own default is 2, which would
# confound the dense-vs-chunked comparison with twice the gradient steps.
#
# `launch` follows the job's logs and blocks, so each submission is backgrounded under `timeout` --
# the Beaker job keeps running after the follower is killed (same pattern as the ctc_suite fan-out).
#
# Usage:
#   bash src/scripts/train/memexpress/singletask_ladder/launch_outlier_datascale_sweep.sh
#   DRY=1 bash .../launch_outlier_datascale_sweep.sh              # dry_run, no submit
#   ARMS=sparselandmark bash .../launch_outlier_datascale_sweep.sh # one arm only
#   FRACTIONS="1.0 0.5" bash .../launch_outlier_datascale_sweep.sh # subset of scales
set -uo pipefail

REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
SCRIPT_3V=src/scripts/train/memexpress/singletask_ladder/Qwen3-4B-singletask-ladder-32k-10k-3variant-SFT.py
SCRIPT_DC=src/scripts/train/memexpress/singletask_ladder/Qwen3-4B-docchunk-singletask-ladder-10k-SFT.py
CLUSTER="${CLUSTER:-ai2/jupiter-cirrascale-2}"
CMD="launch"; [ "${DRY:-0}" = "1" ] && CMD="dry_run"
TASK=outlier
ARMS="${ARMS:-docchunk sparselandmark}"
# fraction:suffix -- suffixes match the dense sweep's run names exactly.
FRACTIONS="${FRACTIONS:-0.025 0.05 0.1 0.25 0.5 1.0}"
LOGD="${LOGD:-debug/outlier_datascale_sweep}"

suffix_for() {
  case "$1" in
    0.025) echo "ss2p5" ;; 0.05) echo "ss5" ;; 0.1) echo "ss10" ;;
    0.25)  echo "ss25"  ;; 0.5)  echo "ss50" ;; 1.0) echo "ss100" ;;
    *) echo "ERROR: no run-name suffix defined for fraction $1" >&2; exit 2 ;;
  esac
}

cd "$REPO" || exit 1
export PYTHONPATH="$REPO/src"
mkdir -p "$LOGD"

echo "=== outlier data-scale sweep (cmd=$CMD, cluster=$CLUSTER) arms=[$ARMS] fractions=[$FRACTIONS] ==="
n=0
for arm in $ARMS; do
  for frac in $FRACTIONS; do
    ss=$(suffix_for "$frac") || exit 2
    if [ "$arm" = "docchunk" ]; then
      RUN_NAME="q4b-docchunk_dense-${TASK}-ladder32k-${ss}"
      SCRIPT="$SCRIPT_DC"
      EXTRA_ENV="STL_EPOCHS=1"
    else
      RUN_NAME="q4b-sparselandmark-${TASK}-ladder32k-${ss}"
      SCRIPT="$SCRIPT_3V"
      EXTRA_ENV=""  # the 3-variant launcher is already EPOCHS=1
    fi
    n=$((n+1))
    LOG="$LOGD/launch_${arm}_${ss}.log"
    echo "--- [$n] $arm / subsample=$frac -> $RUN_NAME (log: $LOG) ---"
    if [ "$CMD" = "dry_run" ]; then
      env STL_SUBSAMPLE="$frac" $EXTRA_ENV python "$SCRIPT" dry_run "$RUN_NAME" "$CLUSTER" \
        > "$LOG" 2>&1
      echo "    dry_run rc=$? (see $LOG)"
    else
      # `launch` blocks streaming logs; background it and kill the follower after registration.
      nohup env STL_SUBSAMPLE="$frac" $EXTRA_ENV timeout 300 \
        python "$SCRIPT" launch "$RUN_NAME" "$CLUSTER" > "$LOG" 2>&1 &
      sleep 5
    fi
  done
done

[ "$CMD" = "dry_run" ] && { echo "=== done: $n dry runs ==="; exit 0; }

echo "waiting for gantry to register experiments..."
sleep 180
pkill -f "Qwen3-4B-docchunk-singletask-ladder-10k-SFT.py" 2>/dev/null
pkill -f "Qwen3-4B-singletask-ladder-32k-10k-3variant-SFT.py" 2>/dev/null
sleep 3
echo "=== EXPERIMENT IDS ==="
ok=0; fail=""
for f in "$LOGD"/launch_*.log; do
  base=$(basename "$f" .log)
  id=$(grep -aoE "https://beaker[^ ]*experiments/[0-9A-Za-z]+" "$f" 2>/dev/null | head -1)
  if [ -n "$id" ]; then echo "$base -> $id"; ok=$((ok+1)); else echo "$base -> NO_ID (check $f)"; fail="$fail $base"; fi
done
echo "=== SUMMARY: $ok/$n launched OK; FAILED:$fail ==="
