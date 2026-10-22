#!/bin/bash
# The non-dense arms of olmo3_cptdrop on one 8-GPU node, in order, stopping at the first failure.
# Budgets are matched to dense 120 / 240 / 480 steps by the smoke FLOP meter (actual/dense per step:
# IRAND 0.184, IFL 0.223, C phase-1 8k 0.89). Each drop arm = one WSD run to its top budget with full
# checkpoints every 0.9 x smallest budget, then decay branches to the two smaller budgets.
#   STAGES="IRAND IFL C" CODE_SHA=<sha> sbatch -w cubbins research/token_routing/olmo3_cptdrop/driver_rest.sbatch
#SBATCH --job-name=o3cd-rest
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=900G
#SBATCH --time=14:00:00
#SBATCH --output=/data/prasann/olmo3_cptdrop/logs/%x-%j.out
set -uo pipefail
ROOT=/data/prasann/olmo3_cptdrop
CODE_SHA="${CODE_SHA:?}"
ARM_SH="$ROOT/code_$CODE_SHA/research/token_routing/olmo3_cptdrop/run_arm.sbatch"
export CODE_SHA
log() { echo "[driver $(date '+%F %T')] $*"; }
run() {
  local name=$1; shift
  if [ -f "$ROOT/runs/$name/model_and_optim/.metadata" ]; then log "SKIP $name (final export exists)"; return 0; fi
  log "START $name"; env "$@" bash "$ARM_SH" > "$ROOT/logs/$name.log" 2>&1; local rc=$?
  log "END $name rc=$rc"; return $rc
}
for st in ${STAGES:-IRAND IFL C}; do
  case $st in
    IRAND)
      run o3cd-IRAND-2600 ARM=IRAND RUN=o3cd-IRAND-2600 MAX_STEPS_TOTAL=2600 SAVE_INTERVAL=585 EXPECTED_MIN=170 || exit 30
      run o3cd-IRAND-650 ARM=IRAND RUN=o3cd-IRAND-650 MAX_STEPS_TOTAL=650 BRANCH_FROM="$ROOT/runs/o3cd-IRAND-2600/step585" EXPECTED_MIN=15 || exit 31
      run o3cd-IRAND-1300 ARM=IRAND RUN=o3cd-IRAND-1300 MAX_STEPS_TOTAL=1300 BRANCH_FROM="$ROOT/runs/o3cd-IRAND-2600/step1170" EXPECTED_MIN=20 || exit 32 ;;
    IFL)
      run o3cd-IFL-2160 ARM=IFL RUN=o3cd-IFL-2160 MAX_STEPS_TOTAL=2160 SAVE_INTERVAL=486 EXPECTED_MIN=120 || exit 40
      run o3cd-IFL-540 ARM=IFL RUN=o3cd-IFL-540 MAX_STEPS_TOTAL=540 BRANCH_FROM="$ROOT/runs/o3cd-IFL-2160/step486" EXPECTED_MIN=12 || exit 41
      run o3cd-IFL-1080 ARM=IFL RUN=o3cd-IFL-1080 MAX_STEPS_TOTAL=1080 BRANCH_FROM="$ROOT/runs/o3cd-IFL-2160/step972" EXPECTED_MIN=15 || exit 42 ;;
    C)
      for B in 120 240 480; do
        H=$((B / 2))
        run "o3cd-C-$B-p1" ARM=C1 RUN="o3cd-C-$B-p1" MAX_STEPS_TOTAL=$B HARD_STOP=$H EXPECTED_MIN=$((H / 5 + 10)) || exit 50
        run "o3cd-C-$B" ARM=C2 RUN="o3cd-C-$B" MAX_STEPS_TOTAL=$H BRANCH_FROM="$ROOT/runs/o3cd-C-$B-p1/step$H" \
          BRANCH_OPTIM_ONLY=1 WARMUP=0 DECAY_FRAC=0.2 EXPECTED_MIN=$((H / 5 + 10)) || exit 51
      done ;;
  esac
done
log "REST DONE"
