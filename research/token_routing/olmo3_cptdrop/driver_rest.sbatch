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
CODE="$ROOT/code_$CODE_SHA"
ARM_SH="$CODE/research/token_routing/olmo3_cptdrop/run_arm.sbatch"
PY=/data/prasann/conda/envs/corpus-reasoning-olmo/bin/python
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
      # save every 195: 585 / 1170 are the 650 / 1300 branch points; 390 / 585 / 1170 also seed the HO arms
      run o3cd-IRAND-2600 ARM=IRAND RUN=o3cd-IRAND-2600 MAX_STEPS_TOTAL=2600 SAVE_INTERVAL=195 EXPECTED_MIN=190 || exit 30
      run o3cd-IRAND-650 ARM=IRAND RUN=o3cd-IRAND-650 MAX_STEPS_TOTAL=650 BRANCH_FROM="$ROOT/runs/o3cd-IRAND-2600/step585" EXPECTED_MIN=15 || exit 31
      run o3cd-IRAND-1300 ARM=IRAND RUN=o3cd-IRAND-1300 MAX_STEPS_TOTAL=1300 BRANCH_FROM="$ROOT/runs/o3cd-IRAND-2600/step1170" EXPECTED_MIN=20 || exit 32 ;;
    IFL)
      run o3cd-IFL-2160 ARM=IFL RUN=o3cd-IFL-2160 MAX_STEPS_TOTAL=2160 SAVE_INTERVAL=486 EXPECTED_MIN=120 || exit 40
      run o3cd-IFL-540 ARM=IFL RUN=o3cd-IFL-540 MAX_STEPS_TOTAL=540 BRANCH_FROM="$ROOT/runs/o3cd-IFL-2160/step486" EXPECTED_MIN=12 || exit 41
      run o3cd-IFL-1080 ARM=IFL RUN=o3cd-IFL-1080 MAX_STEPS_TOTAL=1080 BRANCH_FROM="$ROOT/runs/o3cd-IFL-2160/step972" EXPECTED_MIN=15 || exit 42 ;;
    DFIX)  # the dense driver lost D-480/step108 (max_checkpoints=3): D-240 branch from step216, D-120 fresh x2 seeds
      run o3cd-D-240 ARM=D RUN=o3cd-D-240 MAX_STEPS_TOTAL=240 BRANCH_FROM="$ROOT/runs/o3cd-D-480/step216" EXPECTED_MIN=15 || exit 70
      run o3cd-D-120 ARM=D RUN=o3cd-D-120 MAX_STEPS_TOTAL=120 EXPECTED_MIN=30 || exit 71
      run o3cd-D-120-s1 ARM=D RUN=o3cd-D-120-s1 MAX_STEPS_TOTAL=120 SEED=1 EXPECTED_MIN=30 || exit 72 ;;
    LRPROBE)  # dense plateaued at 2.07e-4 / 512k-token batches: probe lower peak LRs (240 steps + 120 branch each)
      "$PY" -u "$ROOT/s3_sync.py" down "$ROOT/base/stage3-step11921" base/stage3-step11921 | tail -1   # eval reference
      for pair in "6.5e-5:lr65" "2e-5:lr20"; do
        lr=${pair%%:*}; tag=${pair##*:}
        run "o3cd-D$tag-240" ARM=D LR=$lr RUN="o3cd-D$tag-240" MAX_STEPS_TOTAL=240 SAVE_INTERVAL=108 EXPECTED_MIN=60 || exit 80
        run "o3cd-D$tag-120" ARM=D LR=$lr RUN="o3cd-D$tag-120" MAX_STEPS_TOTAL=120 BRANCH_FROM="$ROOT/runs/o3cd-D$tag-240/step108" EXPECTED_MIN=10 || exit 81
      done
      SPECS="Dlr65-120:runs/o3cd-Dlr65-120:4 Dlr65-240:runs/o3cd-Dlr65-240:4 Dlr20-120:runs/o3cd-Dlr20-120:4 Dlr20-240:runs/o3cd-Dlr20-240:4 stage3-yarn8:base/stage3-step11921:8" \
      SHARDS="dev32k_1024:128 short4k_v2:256" CODE="$CODE" PY="$PY" ROOT="$ROOT" \
        bash "$CODE/research/token_routing/olmo3_cptdrop/devce_multi.sh" > "$ROOT/logs/devce-lrprobe-$SLURM_JOB_ID.log" 2>&1 \
        || log "WARN: LR-probe eval had failures"
      grep -h "merged" "$ROOT/logs/devce-lrprobe-$SLURM_JOB_ID.log" ;;
    EVALD)  # dense family + the CPT start, 8-GPU row-chunked scoring
      for d in dev32k_1024 short4k_v2; do \
        "$PY" -u "$ROOT/s3_sync.py" down "$ROOT/data/$d" "data/$d" | tail -1; done
      SPECS="stage2-yarn4:base/stage2-step47684:4 D-120:runs/o3cd-D-120:4 D-240:runs/o3cd-D-240:4 D-480:runs/o3cd-D-480:4 D-120-s1:runs/o3cd-D-120-s1:4" \
      SHARDS="dev32k_1024:128 short4k_v2:256" CODE="$CODE" PY="$PY" ROOT="$ROOT" \
        bash "$CODE/research/token_routing/olmo3_cptdrop/devce_multi.sh" > "$ROOT/logs/devce-dense-$SLURM_JOB_ID.log" 2>&1 \
        || log "WARN: dense eval had failures"
      grep -h "merged" "$ROOT/logs/devce-dense-$SLURM_JOB_ID.log" ;;
    HO)  # handoff: IRAND for the first stretch, then DENSE to the budget (trainer state kept: data continues)
      for spec in "390:120" "585:240" "1170:480"; do
        S0=${spec%%:*}; B=${spec##*:}
        # dense-equivalent FLOPs so far = S0 x 0.184; dense steps to reach budget B
        DS=$(python3 -c "print(round($B - $S0 * 0.184))")
        run "o3cd-HO-$B" ARM=D RUN="o3cd-HO-$B" MAX_STEPS_TOTAL=$((S0 + DS)) BRANCH_FROM="$ROOT/runs/o3cd-IRAND-2600/step$S0" EXPECTED_MIN=$((DS / 4 + 10)) || exit 60
      done ;;
    EVALI)
      SPECS="IRAND-650:runs/o3cd-IRAND-650:4 IRAND-1300:runs/o3cd-IRAND-1300:4 IRAND-2600:runs/o3cd-IRAND-2600:4 HO-120:runs/o3cd-HO-120:4 HO-240:runs/o3cd-HO-240:4 HO-480:runs/o3cd-HO-480:4" \
      SHARDS="dev32k_1024:128 short4k_v2:256" CODE="$CODE" PY="$PY" ROOT="$ROOT" \
        bash "$CODE/research/token_routing/olmo3_cptdrop/devce_multi.sh" > "$ROOT/logs/devce-irand-$SLURM_JOB_ID.log" 2>&1 \
        || log "WARN: IRAND/HO eval had failures"
      grep -h "merged" "$ROOT/logs/devce-irand-$SLURM_JOB_ID.log" ;;
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
