#!/bin/bash
# Submit the ON-BEAKER multi-rung native eval for the overnight single-task 10k matrix -- one Beaker
# job per (run, task), fully on Beaker (reads eval code+data+checkpoint from weka, no local sync).
#
# Run AFTER the matching training jobs have produced checkpoints on weka (the on-node runner globs the
# LATEST complete step under checkpoints/prasanns/<run>/). Re-running is safe (eval is idempotent;
# results overwrite). Prereqs: `bash upload_lc_eval_bundle.sh` done once; conda env with beaker creds.
#
# Run-name convention (matches launch_singletask_10k_overnight.sh):
#   q4b-<variant>-<task>-ladder32k-10k            for dense|compressive|landmark
#   q4b-docchunk_dense-<task>-ladder32k-10k       for docchunk (OOLONG only)
#
# Usage:
#   bash src/scripts/train/sft/singletask_ladder/launch_beaker_multirung_eval.sh                       # all variants x tasks
#   DRY=1 bash .../launch_beaker_multirung_eval.sh                                   # build, don't submit
#   CLUSTER=ai2/jupiter bash .../launch_beaker_multirung_eval.sh                     # override cluster
#   VARIANTS="dense compressive landmark" TASKS="contra nq" bash .../launch_...sh    # subset
set -uo pipefail

REPO="${REPO:-/accounts/projects/berkeleynlp/prasann/projects/OLMo-core}"
LAUNCHER="$REPO/src/scripts/train/sft/singletask_ladder/run_q4b_beaker_multirung_eval.py"
CLUSTER="${CLUSTER:-ai2/jupiter}"
TASKS="${TASKS:-contra nq rerank outlier oolong}"
VARIANTS="${VARIANTS:-dense compressive landmark docchunk}"
PRIORITY="${PRIORITY:-normal}"
DRY_FLAG=""; [ "${DRY:-0}" = "1" ] && DRY_FLAG="--dry-run"

cd "$REPO"
export PYTHONPATH="$REPO/src"

echo "=== Beaker eval matrix (cluster=$CLUSTER dry=${DRY:-0}) variants=[$VARIANTS] tasks=[$TASKS] ==="
n=0
for variant in $VARIANTS; do
  for task in $TASKS; do
    if [ "$variant" = "docchunk" ]; then
      [ "$task" != "oolong" ] && continue          # docchunk native eval = OOLONG only
      RUN_NAME="q4b-docchunk_dense-${task}-ladder32k-10k"; VFLAG="--variant docchunk"
    else
      RUN_NAME="q4b-${variant}-${task}-ladder32k-10k"; VFLAG="--variant $variant"
    fi
    n=$((n+1))
    echo "--- [$n] $variant / $task -> $RUN_NAME ---"
    python "$LAUNCHER" "$RUN_NAME" "$CLUSTER" --task "$task" $VFLAG --priority "$PRIORITY" $DRY_FLAG
  done
done
echo "=== done: $n eval jobs (cluster=$CLUSTER dry=${DRY:-0}) ==="
