#!/bin/bash
# One 8-GPU allocation (cubbins) that runs, in order, stopping at the first failure:
#   1. unit tests (1 GPU): position window, positional soft tokens, soft-token regressions, dev-CE CPU tests
#   2. smoke (no wandb, no exports): D 4 steps w/ a step-3 checkpoint; decay branch from it (trainer state);
#      C2-style optimizer-only branch from it; C1 (8k) 3 steps; IRAND 3 steps; IFL 3 steps.
#      Automatic check per smoke: exit 0, step lines present, CE finite, FLOP meter > 0.
#   3. dense family: o3cd-D-480 (save 108/216/324/432) -> branches o3cd-D-120 (from step108) and
#      o3cd-D-240 (from step216) -> second seed o3cd-D-120-s1.
#   sbatch -w cubbins research/token_routing/olmo3_cptdrop/driver_dense.sbatch   (CODE_SHA in env)
#SBATCH --job-name=o3cd-dense
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
ENV=/data/prasann/conda/envs/corpus-reasoning-olmo
J=$SLURM_JOB_ID
export CODE_SHA
log() { echo "[driver $(date '+%F %T')] $*"; }

check_smoke() {  # $1 = run log
  local f=$1
  grep -qE "train/CE loss" "$f" || { log "SMOKE FAIL: no CE lines in $f"; return 1; }
  if grep -E "train/CE loss" "$f" | grep -qiE "nan|inf"; then log "SMOKE FAIL: non-finite CE in $f"; return 1; fi
  grep -qE "flop_meter/actual_pflops" "$f" || log "WARN: no flop_meter lines on the console in $f (check wandb/metrics files)"
  log "smoke ok: $(grep -E 'train/CE loss' "$f" | tail -1 | tr -s ' ' | cut -c1-160)"
}
run() {  # ARM-specific env passed by caller; $1 = log name
  local name=$1; shift
  log "START $name"
  local tmo; tmo=$(env "$@" bash -c 'echo ${TIMEOUT_MIN:-0}')
  if [ "$tmo" != "0" ]; then timeout "${tmo}m" env "$@" bash "$ARM_SH" > "$ROOT/logs/$name.log" 2>&1
  else env "$@" bash "$ARM_SH" > "$ROOT/logs/$name.log" 2>&1; fi
  local rc=$?
  log "END $name rc=$rc"
  return $rc
}

# 1. unit tests (SKIP_TESTS=1 when a previous driver already passed them on this code)
if [ -z "${SKIP_TESTS:-}" ]; then
log "pytest"
( cd "$CODE" && export PYTHONPATH="$CODE/src:$CODE/recipes/train/research_trainer:$CODE/recipes/train:$CODE" \
    TMPDIR="$ROOT/tmp" TRITON_CACHE_DIR="$ROOT/tmp/triton" CUDA_VISIBLE_DEVICES=0 && \
  "$ENV/bin/python" -m pytest -q -p no:cacheprovider \
    src/test/nn/attention/position_window_test.py src/test/nn/pooled_soft_token_positional_test.py \
    src/test/nn/pooled_soft_token_test.py src/test/nn/pooled_soft_token_first_frac_noslot_test.py \
    src/test/nn/pooled_soft_token_keep_token_test.py src/test/nn/pooled_soft_token_packed_test.py \
    src/test/nn/block_skip_test.py research/token_routing/olmo3_cptdrop/test_eval_devce_cpu.py \
    > "$ROOT/logs/pytest-$J.log" 2>&1 )
rc=$?; tail -5 "$ROOT/logs/pytest-$J.log"; [ $rc -eq 0 ] || { log "pytest FAILED rc=$rc"; exit 10; }
fi

# 2. smoke. SMOKE_FROM=C1 skips D / decay-branch / optim-only-branch smokes already passed on this code
#    (driver 4016772: all three ok). An empty-epoch livelock (no instance fits --seq-len) fails fast below.
S=o3cd-smoke-$J
if [ "${SMOKE_FROM:-D}" = "D" ]; then
# NOT SMOKE=1 here: --no-final-checkpoint disables the whole checkpointer, mid-run saves included, and the
# branch smokes below need step3. Its checkpoints + export are deleted after the smoke.
run "$S-D" ARM=D RUN="$S-D" MAX_STEPS_TOTAL=10 HARD_STOP=4 SAVE_INTERVAL=3 NOWANDB=1 || exit 11
check_smoke "$ROOT/logs/$S-D.log" || exit 11
# budgets must give WSD >= 1 decay step (round(t_max * frac) = 0 divides by zero in the scheduler)
run "$S-Dbr" ARM=D RUN="$S-Dbr" MAX_STEPS_TOTAL=10 BRANCH_FROM="$ROOT/runs/$S-D/step3" SMOKE=1 NOWANDB=1 || exit 12
check_smoke "$ROOT/logs/$S-Dbr.log" || exit 12
grep -q "Will resume training from step 3" "$ROOT/logs/$S-Dbr.log" || { log "SMOKE FAIL: decay branch did not resume from step 3"; exit 12; }
run "$S-C2" ARM=C2 RUN="$S-C2" MAX_STEPS_TOTAL=5 BRANCH_FROM="$ROOT/runs/$S-D/step3" BRANCH_OPTIM_ONLY=1 WARMUP=0 DECAY_FRAC=0.2 SMOKE=1 NOWANDB=1 || exit 13
check_smoke "$ROOT/logs/$S-C2.log" || exit 13
grep -q "Will resume training from step" "$ROOT/logs/$S-C2.log" && { log "SMOKE FAIL: optim-only branch restored trainer state"; exit 13; }
fi
run "$S-C1" ARM=C1 RUN="$S-C1" MAX_STEPS_TOTAL=10 HARD_STOP=3 SMOKE=1 NOWANDB=1 TIMEOUT_MIN=20 || exit 14
check_smoke "$ROOT/logs/$S-C1.log" || exit 14
run "$S-IRAND" ARM=IRAND RUN="$S-IRAND" MAX_STEPS_TOTAL=10 HARD_STOP=3 SMOKE=1 NOWANDB=1 TIMEOUT_MIN=20 || exit 15
check_smoke "$ROOT/logs/$S-IRAND.log" || exit 15
run "$S-IFL" ARM=IFL RUN="$S-IFL" MAX_STEPS_TOTAL=10 HARD_STOP=3 SMOKE=1 NOWANDB=1 TIMEOUT_MIN=20 || exit 16
check_smoke "$ROOT/logs/$S-IFL.log" || exit 16
rm -rf "$ROOT"/runs/o3cd-smoke-*   # smoke checkpoints + exports (~200 GB per dense smoke)
log "SMOKE PASSED"

# 3. dense family
run o3cd-D-480 ARM=D RUN=o3cd-D-480 MAX_STEPS_TOTAL=480 SAVE_INTERVAL=108 EXPECTED_MIN=150 || exit 20
run o3cd-D-120 ARM=D RUN=o3cd-D-120 MAX_STEPS_TOTAL=120 BRANCH_FROM="$ROOT/runs/o3cd-D-480/step108" EXPECTED_MIN=15 || exit 21
run o3cd-D-240 ARM=D RUN=o3cd-D-240 MAX_STEPS_TOTAL=240 BRANCH_FROM="$ROOT/runs/o3cd-D-480/step216" EXPECTED_MIN=15 || exit 22
run o3cd-D-120-s1 ARM=D RUN=o3cd-D-120-s1 MAX_STEPS_TOTAL=120 SEED=1 EXPECTED_MIN=40 || exit 23
log "DENSE FAMILY DONE"
