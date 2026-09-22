#!/bin/bash
# Submit one dev-loss eval per FINISHED sdcpt run that has no result yet (idempotent; rerun until all 12 are in).
# Checkpoint completeness = weka model_and_optim/.metadata, checked from inside a tiny gantry probe is
# too slow, so this trusts the LAUNCH_LEDGER arms and lets eval_cpt_devloss_beaker.sh exit 2 if the
# checkpoint is not there yet (the ledger note records the eval experiment id when it was accepted).
#   bash src/scripts/train/memexpress/cpt/softdetach/eval_sweep.sh
set -uo pipefail
cd "$(dirname "$0")/../../../../../.."
WEKA=/weka/oe-training-default/ai2-llm/checkpoints/prasanns
for arm in dense sd20 sfl20 lslot20; do for b in ${BUDGETS:-32M 64M 128M}; do
  RUN=sdcpt-q35-4b-$arm-u$b
  grep -q "^EVAL	$RUN	" src/scripts/train/memexpress/cpt/softdetach/EVAL_LEDGER.tsv 2>/dev/null && continue
  OUT=$(RUN=$RUN ARM=$arm bash src/scripts/train/memexpress/cpt/softdetach/eval_cpt_devloss_beaker.sh 2>&1 | grep -oE "ex/[A-Z0-9]{26}" | head -1 | cut -d/ -f2)
  [ -n "$OUT" ] && printf "EVAL\t%s\t%s\t%s\t%s\n" "$RUN" "$arm" "$OUT" "$(date '+%Y-%m-%d %H:%M')" >> src/scripts/train/memexpress/cpt/softdetach/EVAL_LEDGER.tsv
  echo "$RUN -> ${OUT:-submit failed}"
done; done
