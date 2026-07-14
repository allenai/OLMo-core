#!/bin/bash
# Re-run any gold-grad arm that finished WITHOUT a saved checkpoint.
#
# Why: at seq 6144 (the n100 rung) flash-attn's backward intermittently SIGSEGVs (exitcode -11) partway
# through an arm -- at 33/62 steps on one node, 60/63 on another, i.e. random, not a fixed trigger. It
# is NOT the gold-grad detach: the `full` arm installs no hook at all and still crashed. The n20 rung
# (seq 2048) has never hit it. So: retry, don't redesign.
#
#   bash src/scripts/train/goldgrad/reap_q06b_goldgrad_local.sh n100 [max_retries]
set -uo pipefail
RUNG="${1:-n100}"
MAX="${2:-2}"
REPO="${REPO:-/accounts/projects/berkeleynlp/prasann/projects/OLMo-core}"
TAG="${TAG:-famark}"
SAVE_ROOT="${SAVE_ROOT:-/data/prasann/olmo_ckpts}"
ARMS="${ARMS:-gold_plus_random:0:2:gpr2 random_only:0:2:rand2 gold_subsample:1:15:gsub1_15}"

for attempt in $(seq 1 "$MAX"); do
  missing=""
  for arm in $ARMS; do
    IFS=: read -r MODE NGOLD NRAND SUF <<< "$arm"
    RUN="q06b-goldgrad-${TAG}-${RUNG}-${SUF}"
    [ -f "$SAVE_ROOT/$RUN/model_and_optim/.metadata" ] || missing="$missing $arm"
  done
  missing="${missing# }"
  if [ -z "$missing" ]; then echo "=== all $RUNG arms have checkpoints ==="; exit 0; fi
  echo "=== retry pass $attempt/$MAX -- missing: $missing ==="
  ARMS="$missing" bash "$REPO/src/scripts/train/goldgrad/run_q06b_goldgrad_local.sh" "$RUNG"
done

echo "=== after $MAX retries, still missing: ==="
for arm in $ARMS; do
  IFS=: read -r MODE NGOLD NRAND SUF <<< "$arm"
  RUN="q06b-goldgrad-${TAG}-${RUNG}-${SUF}"
  [ -f "$SAVE_ROOT/$RUN/model_and_optim/.metadata" ] || echo "  MISSING $RUN"
done
