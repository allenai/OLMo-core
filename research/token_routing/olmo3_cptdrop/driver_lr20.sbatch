#!/bin/bash
# olmo3_cptdrop at peak LR 2e-5 (the LR probe: 2e-5 beats the OLMo recipe's 2.07e-4 by -0.133 +-0.002 nats at
# 120 steps and keeps improving with compute; 2.07e-4 plateaus and forgets). One 8-GPU node, in order, stopping at
# the first failure; runs whose export exists are skipped (rerunnable).
#   DC  dense curve: Dlr20-{12,24,48} fresh WSD runs (+ Dlr20-120/240 already from LRPROBE), Dlr10-120 (LR 1e-5)
#   IR  IRANDlr20-2600 (saves every 195) + decay branches 650 / 1300
#   HO  HOlr20-{120,240,480}: IRAND for 390 / 585 / 1170 steps, then dense to the budget (trainer state kept)
#   SB  dense small-batch control DB8lr20-{12,24,48}: 8 rows/step at the same token budget = 2x the optimizer
#       steps (do drop arms win by dropping, or only by taking more optimizer steps per FLOP?)
#   SEED Dlr20-120-s1 (seed noise of the dense reference at LR 2e-5; was 0.009 nats at 2.07e-4)
#   IRS  IRANDlr20-260 (saves every 58) + branches 65 / 130: I-rand at dense-equivalent 12 / 24 / 48 steps
#   IFS  IFLlr20-215 (saves every 48) + branches 54 / 108: I-fl 25% at dense-equivalent 12 / 24 / 48 steps
#   D480 Dlr20-480 fresh (the long end of the dense curve)
#   EV  8-GPU row-chunked dev CE of everything above
#   STAGES="DC IR HO EV" CODE_SHA=<sha> sbatch -w cubbins research/token_routing/olmo3_cptdrop/driver_lr20.sbatch
#SBATCH --job-name=o3cd-lr20
#SBATCH --partition=jsteinhardt
#SBATCH --qos=preemptive_high
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=900G
#SBATCH --time=16:00:00
#SBATCH --output=/data/prasann/olmo3_cptdrop/logs/%x-%j.out
set -uo pipefail
ROOT=/data/prasann/olmo3_cptdrop
CODE_SHA="${CODE_SHA:?}"
CODE="$ROOT/code_$CODE_SHA"
ARM_SH="$CODE/research/token_routing/olmo3_cptdrop/run_arm.sbatch"
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
export CODE_SHA LR=2e-5
log() { echo "[driver $(date '+%F %T')] $*"; }
run() {
  local name=$1; shift
  if [ -f "$ROOT/runs/$name/model_and_optim/.metadata" ]; then log "SKIP $name (export exists)"; return 0; fi
  log "START $name"; env "$@" RUN="$name" bash "$ARM_SH" > "$ROOT/logs/$name.log" 2>&1; local rc=$?
  log "END $name rc=$rc"; return $rc
}
for st in ${STAGES:-DC SB IRS IFS EV SEED IR HO D480 EV}; do
  case $st in
    DC)
      for b in 12 24 48; do run "o3cd-Dlr20-$b" ARM=D MAX_STEPS_TOTAL=$b EXPECTED_MIN=$((b / 4 + 8)) || exit 10; done
      run o3cd-Dlr10-120 ARM=D LR=1e-5 MAX_STEPS_TOTAL=120 EXPECTED_MIN=30 || exit 11 ;;
    SB)
      for b in 12 24 48; do run "o3cd-DB8lr20-$b" ARM=D GB_OVERRIDE=8 MAX_STEPS_TOTAL=$b EXPECTED_MIN=$((b / 3 + 8)) || exit 13; done ;;
    SEED)
      run o3cd-Dlr20-120-s1 ARM=D MAX_STEPS_TOTAL=120 SEED=1 EXPECTED_MIN=30 || exit 14 ;;
    IRS)
      run o3cd-IRANDlr20-260 ARM=IRAND MAX_STEPS_TOTAL=260 SAVE_INTERVAL=58 EXPECTED_MIN=30 || exit 23
      run o3cd-IRANDlr20-65 ARM=IRAND MAX_STEPS_TOTAL=65 BRANCH_FROM="$ROOT/runs/o3cd-IRANDlr20-260/step58" EXPECTED_MIN=8 || exit 24
      run o3cd-IRANDlr20-130 ARM=IRAND MAX_STEPS_TOTAL=130 BRANCH_FROM="$ROOT/runs/o3cd-IRANDlr20-260/step116" EXPECTED_MIN=8 || exit 25 ;;
    IFS)
      run o3cd-IFLlr20-215 ARM=IFL MAX_STEPS_TOTAL=215 SAVE_INTERVAL=48 EXPECTED_MIN=30 || exit 26
      run o3cd-IFLlr20-54 ARM=IFL MAX_STEPS_TOTAL=54 BRANCH_FROM="$ROOT/runs/o3cd-IFLlr20-215/step48" EXPECTED_MIN=8 || exit 27
      run o3cd-IFLlr20-108 ARM=IFL MAX_STEPS_TOTAL=108 BRANCH_FROM="$ROOT/runs/o3cd-IFLlr20-215/step96" EXPECTED_MIN=8 || exit 28 ;;
    D480)
      run o3cd-Dlr20-480 ARM=D MAX_STEPS_TOTAL=480 EXPECTED_MIN=110 || exit 12 ;;
    IR)
      run o3cd-IRANDlr20-2600 ARM=IRAND MAX_STEPS_TOTAL=2600 SAVE_INTERVAL=195 EXPECTED_MIN=190 || exit 20
      run o3cd-IRANDlr20-650 ARM=IRAND MAX_STEPS_TOTAL=650 BRANCH_FROM="$ROOT/runs/o3cd-IRANDlr20-2600/step585" EXPECTED_MIN=15 || exit 21
      run o3cd-IRANDlr20-1300 ARM=IRAND MAX_STEPS_TOTAL=1300 BRANCH_FROM="$ROOT/runs/o3cd-IRANDlr20-2600/step1170" EXPECTED_MIN=20 || exit 22 ;;
    HO)
      for spec in "390:120" "585:240" "1170:480"; do
        S0=${spec%%:*}; B=${spec##*:}
        DS=$(python3 -c "print(round($B - $S0 * 0.184))")
        run "o3cd-HOlr20-$B" ARM=D MAX_STEPS_TOTAL=$((S0 + DS)) BRANCH_FROM="$ROOT/runs/o3cd-IRANDlr20-2600/step$S0" EXPECTED_MIN=$((DS / 4 + 10)) || exit 30
      done ;;
    EV)
      SPECS=""
      for r in Dlr20-12 Dlr20-24 Dlr20-48 DB8lr20-12 DB8lr20-24 DB8lr20-48 Dlr20-120-s1 IRANDlr20-65 IRANDlr20-130 IRANDlr20-260 IFLlr20-54 IFLlr20-108 IFLlr20-215 Dlr10-120 Dlr20-480 IRANDlr20-650 IRANDlr20-1300 IRANDlr20-2600 HOlr20-120 HOlr20-240 HOlr20-480; do
        [ -f "$ROOT/runs/o3cd-$r/model_and_optim/.metadata" ] && [ ! -f "$ROOT/devce/${r}_dev32k_1024.json" ] && SPECS="$SPECS $r:runs/o3cd-$r:4"
      done
      log "EV specs:$SPECS"
      SPECS="$SPECS" SHARDS="dev32k_1024:128 short4k_v2:256" CODE="$CODE" PY="$PY" ROOT="$ROOT" \
        bash "$CODE/research/token_routing/olmo3_cptdrop/devce_multi.sh" > "$ROOT/logs/devce-lr20-$SLURM_JOB_ID.log" 2>&1 \
        || log "WARN: eval had failures"
      grep -h "merged" "$ROOT/logs/devce-lr20-$SLURM_JOB_ID.log" ;;
  esac
done
log "LR20 DONE"
