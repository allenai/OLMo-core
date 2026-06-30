#!/bin/bash
# Submit all 15 Beaker fine-tune jobs: 5 tasks x 3 attention variants, each on a 50% (~10k of 20k)
# subsample of one task's single-task length ladder, NO CPT mix.
#
# PREPARE-ONLY: this script SELF-SUBMITS to Beaker via the olmo-core `launch` command. Do NOT run it
# until the small-scale results are in AND the weka data upload under
# prasanns/single_task_ladders/<task>/ is confirmed complete.
#
# Prereqs:
#   * conda env `corpus-reasoning-olmo` active (gantry + beaker creds live there), `beaker account whoami` ok.
#   * weka data present:  /weka/oe-training-default/ai2-llm/checkpoints/prasanns/single_task_ladders/<task>/
#   * weka CPT bases present under .../checkpoints/amandab/ (dense/landmark/compressive -- see launcher).
#
# Usage:
#   bash src/scripts/train/sft/singletask_ladder/launch_singletask_10k_3variant.sh                 # submit all 15
#   CLUSTER=ai2/jupiter bash .../launch_singletask_10k_3variant.sh               # override cluster
#   DRY=1 bash .../launch_singletask_10k_3variant.sh                             # dry_run (no submit)
#   TASKS="contra nq" VARIANTS="dense landmark" bash .../launch_...sh            # subset
set -uo pipefail

REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
SCRIPT=src/scripts/train/sft/singletask_ladder/Qwen3-4B-singletask-ladder-32k-10k-3variant-SFT.py
CLUSTER="${CLUSTER:-ai2/neptune}"
CMD="launch"; [ "${DRY:-0}" = "1" ] && CMD="dry_run"
TASKS="${TASKS:-contra nq oolong rerank outlier}"
VARIANTS="${VARIANTS:-dense landmark compressive}"

cd "$REPO"
export PYTHONPATH="$REPO/src"

echo "=== submitting (cmd=$CMD, cluster=$CLUSTER) tasks=[$TASKS] variants=[$VARIANTS] ==="
n=0
for task in $TASKS; do
  for variant in $VARIANTS; do
    # run name must contain BOTH the variant keyword and the task keyword (launcher parses both).
    RUN_NAME="q4b-${variant}-${task}-ladder32k-10k"
    n=$((n+1))
    echo "--- [$n] $RUN_NAME ---"
    python "$SCRIPT" "$CMD" "$RUN_NAME" "$CLUSTER"
  done
done
echo "=== done: $n jobs ($CMD) ==="
