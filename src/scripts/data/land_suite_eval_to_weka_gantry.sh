#!/usr/bin/env bash
# Land the corpus-reasoning task-suite EVAL JSONL on weka via a Beaker CPU gantry job,
# so the oe-eval `cr_*` suite tasks (oe-eval branch prasann/longctx-eval) can read it.
#
# The data lives in a PUBLIC GCS bucket, so the job just curls it onto the mounted weka
# bucket -- no auth, no Beaker dataset upload. Wraps src/scripts/data/download_suite_data.sh.
#
# Default destination matches DEFAULT_DATA_DIRS in oe-eval
# corpus_reasoning_suite.py: <weka>/ai2-llm/checkpoints/prasanns/cr_suite_eval
# (override that at eval time with CR_SUITE_DATA_DIR if you land it elsewhere).
#
# Usage:
#   src/scripts/data/land_suite_eval_to_weka_gantry.sh                 # all eval files (~5 GB)
#   TASKS="contradiction oolong" src/scripts/data/land_suite_eval_to_weka_gantry.sh   # subset
#   FULL=1 src/scripts/data/land_suite_eval_to_weka_gantry.sh          # eval + train (~22.7 GB)
#
# Overridable via env vars:
#   CLUSTER (ai2/jupiter-cirrascale-2)  WORKSPACE (ai2/flex2)  BUDGET (ai2/oe-other)
#   WEKA (oe-training-default)  PRIORITY (normal)  CPUS (8)  NAME (cr-suite-eval-land)
#   DEST_REL (ai2-llm/checkpoints/prasanns/cr_suite_eval)  TASKS ("")  FULL (0)
#
# NOTE: gantry ships your committed git HEAD -- commit download_suite_data.sh before launching.
set -euo pipefail

CLUSTER="${CLUSTER:-ai2/jupiter-cirrascale-2}"
WORKSPACE="${WORKSPACE:-ai2/flex2}"
BUDGET="${BUDGET:-ai2/oe-other}"
WEKA="${WEKA:-oe-training-default}"
PRIORITY="${PRIORITY:-normal}"
CPUS="${CPUS:-8}"
NAME="${NAME:-cr-suite-eval-land}"
DEST_REL="${DEST_REL:-ai2-llm/checkpoints/prasanns/cr_suite_eval}"

CLUSTER_ARGS=()
IFS=',' read -ra _CLUSTERS <<< "${CLUSTER}"
for c in "${_CLUSTERS[@]}"; do
  CLUSTER_ARGS+=(--cluster "$c")
done

# eval-only by default; FULL=1 grabs the train split too.
SPLIT_FLAG="--eval-only"
[ "${FULL:-0}" = "1" ] && SPLIT_FLAG=""

DEST="/weka/${WEKA}/${DEST_REL}"

gantry run \
  --name "${NAME}" \
  --description "Land corpus-reasoning suite eval JSONL on weka (${DEST_REL})" \
  --workspace "${WORKSPACE}" \
  --budget "${BUDGET}" \
  "${CLUSTER_ARGS[@]}" \
  --weka "${WEKA}:/weka/${WEKA}" \
  --cpus "${CPUS}" \
  --gpus 0 \
  --priority "${PRIORITY}" \
  --allow-dirty \
  --timeout 0 \
  --env "OUT=${DEST}" \
  --env "TASKS=${TASKS:-}" \
  --yes \
  -- bash src/scripts/data/download_suite_data.sh ${SPLIT_FLAG}

echo "Launched ${NAME}: landing suite eval data -> ${DEST}"
echo "At eval time the cr_* tasks read this path by default (or set CR_SUITE_DATA_DIR=${DEST})."
