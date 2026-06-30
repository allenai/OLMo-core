#!/bin/bash
# OVERNIGHT 5-task x 4-variant 10k matrix: submit all 20 Beaker fine-tune jobs (each on a seeded 50%
# ~10k-of-20k subsample of one task's single-task ladder, NO CPT mix, LR 2e-5), in PRIORITY ORDER:
#
#   1. dense        (flash-attn2 + YaRN)                 -> 3-variant launcher
#   2. compressive  (fast_compressive_landmark)          -> 3-variant launcher
#   3. docchunk     (DocumentChunkedAttention, dense)    -> docchunk single-task launcher  [MAY NOT LAUNCH]
#   4. landmark     (fast_landmark)                       -> 3-variant launcher
#
# All 5 tasks of a variant are submitted before moving to the next variant, so the highest-priority
# variants get scheduled first.
#
# !!! PREPARE-ONLY: this SELF-SUBMITS to Beaker via the olmo-core `launch` command. The coordinator
# gates the actual launch on the weka SFT-data upload under
# prasanns/single_task_ladders/<task>/ (dense/compressive/landmark) being complete.
#
# !!! DOCCHUNK FEASIBILITY: the `docchunk` variant reads BOX-MARKER tokenization (it degenerates to
# plain causal attention without <|box_start|>/<|box_end|> markers, which the plain single_task_ladders
# shards do NOT have). It reads DOCCHUNK_DATA_ROOT/<task>_dense instead. CONFIRM that root holds a
# single-task box-marker re-tokenization of the LATEST data (incl. CE-graded rerank) before relying on
# it; otherwise exclude docchunk from VARIANTS (it is intentionally the variant that may not launch).
#
# Prereqs:
#   * conda env `corpus-reasoning-olmo` active (gantry + beaker creds), `beaker account whoami` ok.
#   * weka data present: .../prasanns/single_task_ladders/<task>/  (+ box-marker docchunk root).
#   * weka CPT bases present under .../checkpoints/amandab/ (see launchers).
#
# Usage:
#   bash src/scripts/train/sft/singletask_ladder/launch_singletask_10k_overnight.sh                    # submit all 20
#   DRY=1 bash .../launch_singletask_10k_overnight.sh                                # dry_run (no submit)
#   CLUSTER=ai2/jupiter bash .../launch_singletask_10k_overnight.sh                  # override cluster
#   VARIANTS="dense compressive landmark" bash .../launch_...sh                      # skip docchunk
#   TASKS="contra nq" bash .../launch_...sh                                          # subset of tasks
#   DOCCHUNK_DATA_ROOT=/weka/.../single_task_docchunk bash .../launch_...sh          # re-point box data
set -uo pipefail

REPO=/accounts/projects/berkeleynlp/prasann/projects/OLMo-core
SCRIPT_3V=src/scripts/train/sft/singletask_ladder/Qwen3-4B-singletask-ladder-32k-10k-3variant-SFT.py
SCRIPT_DC=src/scripts/train/sft/singletask_ladder/Qwen3-4B-docchunk-singletask-ladder-10k-SFT.py
CLUSTER="${CLUSTER:-ai2/jupiter}"
CMD="launch"; [ "${DRY:-0}" = "1" ] && CMD="dry_run"
TASKS="${TASKS:-contra nq oolong rerank outlier}"
# Priority order: dense, compressive, docchunk, landmark.
VARIANTS="${VARIANTS:-dense compressive docchunk landmark}"

cd "$REPO"
export PYTHONPATH="$REPO/src"

echo "=== overnight matrix (cmd=$CMD, cluster=$CLUSTER) variants=[$VARIANTS] tasks=[$TASKS] ==="
[ -n "${DOCCHUNK_DATA_ROOT:-}" ] && export DOCCHUNK_DATA_ROOT && echo "    DOCCHUNK_DATA_ROOT=$DOCCHUNK_DATA_ROOT"
n=0
for variant in $VARIANTS; do
  for task in $TASKS; do
    n=$((n+1))
    if [ "$variant" = "docchunk" ]; then
      # docchunk run name must NOT contain dense/landmark/compressive (the 3-variant parser would
      # mis-detect them); use the explicit "docchunk_dense" token + the task keyword.
      RUN_NAME="q4b-docchunk_dense-${task}-ladder32k-10k"
      SCRIPT="$SCRIPT_DC"
    else
      RUN_NAME="q4b-${variant}-${task}-ladder32k-10k"
      SCRIPT="$SCRIPT_3V"
    fi
    echo "--- [$n] $variant / $task -> $RUN_NAME ---"
    python "$SCRIPT" "$CMD" "$RUN_NAME" "$CLUSTER"
  done
done
echo "=== done: $n jobs ($CMD) ==="
