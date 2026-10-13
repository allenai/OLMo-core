#!/bin/bash
# Run a local (Berkeley slurm) job payload under the ops run registry and stall guard.
#
#   bash "$REPO/recipes/env/ops_job.sh" CMD [ARGS...]
#
# 1. registers the run   (ops register: cluster=slurm, job_id=$SLURM_JOB_ID, node, log_path, git)
# 2. runs CMD under      `ops guard` (flags a silent start / a mid-run stall in the job log;
#                         it does NOT kill unless OPS_GUARD_KILL=1)
# 3. records final_state (ops update: done on exit 0, cancelled on SIGINT/SIGTERM/SIGKILL exits,
#                         failed otherwise) and exits with CMD's exit code.
#
# Registry writes are two tiny flock'd appends over NFS. Any ops failure prints one warning and
# the payload runs anyway; ops never changes the job's exit code. A job killed outright (slurm
# time limit, scancel -9) never reaches step 3 -- `ops status` fills that in from sacct.
#
# Knobs (environment):
#   OPS_RUN_ID           run_id = run name = save-folder basename (default: $RUN, else
#                        $SLURM_JOB_NAME-$SLURM_JOB_ID). Evals (OPS_KIND=eval) are their own run:
#                        default <OPS_PARENT_RUN>__ev-<OPS_TASK>-<OPS_EVAL_VARIANT>-<YYYYmmddHHMM>,
#                        exported to the payload as OPS_EVAL_RUN_ID (the native eval drivers write
#                        it into the result's _meta, so `ops result add` files rows under this pass)
#   OPS_EXPERIMENT       ops/experiments/<name> key (default: "unassigned", with a warning)
#   OPS_KIND             train|eval|data|other (default: train)
#   OPS_PARENT_RUN       for evals: the training run scored
#   OPS_CKPT_PATH, OPS_TASK, OPS_ARM, OPS_EVAL_VARIANT, OPS_EXPECTED_MINUTES, OPS_NOTES
#   OPS_LOG_PATH         default: the job's slurm StdOut (scontrol), which should be on /data
#   OPS_FIRST_OUTPUT     seconds before "no output yet" fires (default 600 train/eval, 120 other)
#   OPS_STALL            seconds of silence before "stalled" fires (default 1800 train/eval, 300 other)
#   OPS_GUARD=0          run CMD bare (registry rows only)
#   OPS_GUARD_KILL=1     let the guard kill a stalled CMD (exit 124)
#   OPS_BIN              ops entry point (default: the stable ~prasann/bin/ops)
#
# Adopting it in an existing launcher, e.g. recipes/train/research_trainer/run_ctc_local.sbatch --
# prefix the payload line and say which experiment the run belongs to:
#
#   OPS_RUN_ID="$RUN" OPS_EXPERIMENT="${OPS_EXPERIMENT:-ctc_suite}" OPS_CKPT_PATH="$SAVE_LOCAL" \
#   OPS_TASK="$TASK" OPS_ARM="$VARIANT" \
#   bash "$REPO/recipes/env/ops_job.sh" \
#     "$ENV/bin/python" -m torch.distributed.run --nproc_per_node="$NGPU" ... train_ctc_suite.py ...
#
# An eval pass over that run (its own run_id; parent_run = the trained run), e.g. in
# run_q4b_stl_multirung_eval.sbatch; OPS_RECORD_RESULTS=1 also files the result rows:
#
#   OPS_KIND=eval OPS_PARENT_RUN="$RUN" OPS_TASK="$TASK" OPS_EVAL_VARIANT=native OPS_EXPERIMENT=... \
#   OPS_RECORD_RESULTS=1 bash "$REPO/recipes/env/ops_job.sh" \
#     "$ENV/bin/python" -m torch.distributed.run ... eval_lc_native.py --out ... ...
#
# Data builds: OPS_KIND=data (guard defaults 120 s / 300 s). Lambda has no /accounts share, so
# its jobs cannot reach the registry: register them from the login-side submit script instead.

set -u
[ "$#" -gt 0 ] || { echo "usage: ops_job.sh CMD [ARGS...]" >&2; exit 2; }

OPS="${OPS_BIN:-/accounts/projects/berkeleynlp/prasann/bin/ops}"
_REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
KIND="${OPS_KIND:-train}"
if [ "$KIND" = eval ] && [ -z "${OPS_RUN_ID:-}" ] && [ -n "${OPS_PARENT_RUN:-}" ]; then
  RUN_ID="${OPS_PARENT_RUN}__ev-${OPS_TASK:-${SLURM_JOB_NAME:-eval}}-${OPS_EVAL_VARIANT:-default}-$(date +%Y%m%d%H%M)"
else
  RUN_ID="${OPS_RUN_ID:-${RUN:-${SLURM_JOB_NAME:-job}-${SLURM_JOB_ID:-$$}}}"
fi
[ "$KIND" = eval ] && export OPS_EVAL_RUN_ID="$RUN_ID"
EXPERIMENT="${OPS_EXPERIMENT:-unassigned}"
[ -n "${OPS_EXPERIMENT:-}" ] || echo "[ops] WARNING: OPS_EXPERIMENT unset; registering $RUN_ID under 'unassigned'" >&2
CLUSTER=slurm; [ -n "${SLURM_JOB_ID:-}" ] || CLUSTER=local
LOG_PATH="${OPS_LOG_PATH:-}"
if [ -z "$LOG_PATH" ] && [ -n "${SLURM_JOB_ID:-}" ]; then
  LOG_PATH=$(timeout 10 scontrol show job "$SLURM_JOB_ID" 2>/dev/null | sed -n 's/^ *StdOut=//p' | head -1)
fi
case "$KIND" in train|eval) FIRST=600; STALL=1800 ;; *) FIRST=120; STALL=300 ;; esac
FIRST="${OPS_FIRST_OUTPUT:-$FIRST}"; STALL="${OPS_STALL:-$STALL}"

ops() { timeout 60 python3 "$OPS" "$@"; }

reg=(register --run-id "$RUN_ID" --experiment "$EXPERIMENT" --cluster "$CLUSTER" --kind "$KIND"
     --node "$(hostname -s)" --git "$_REPO" --source sbatch --command "$*")
[ -n "${SLURM_JOB_ID:-}" ] && reg+=(--job-id "$SLURM_JOB_ID")
[ -n "$LOG_PATH" ] && reg+=(--log-path "$LOG_PATH")
[ -n "${OPS_CKPT_PATH:-}" ] && reg+=(--ckpt-path "$OPS_CKPT_PATH")
[ -n "${OPS_PARENT_RUN:-}" ] && reg+=(--parent-run "$OPS_PARENT_RUN")
[ -n "${OPS_TASK:-}" ] && reg+=(--task "$OPS_TASK")
[ -n "${OPS_ARM:-}" ] && reg+=(--arm "$OPS_ARM")
[ -n "${OPS_EXPECTED_MINUTES:-}" ] && reg+=(--expected-minutes "$OPS_EXPECTED_MINUTES")
[ -n "${OPS_NOTES:-}" ] && reg+=(--notes "$OPS_NOTES")
registered=0
if ops "${reg[@]}"; then registered=1; else echo "[ops] WARNING: registry launch row not written; running anyway" >&2; fi

# Survive the scancel/preemption SIGTERM (slurm signals every process in the job, so the payload
# still gets it) long enough to record final_state; slurm's SIGKILL follows after KillWait.
trap 'echo "[ops] job signalled; recording final state after the payload exits" >&2' TERM INT
use_guard=0
if [ "${OPS_GUARD:-1}" != 0 ]; then
  if timeout 60 python3 "$OPS" guard --help >/dev/null 2>&1; then use_guard=1
  else echo "[ops] WARNING: ops guard unavailable; running the payload bare" >&2; fi
fi
if [ "$use_guard" = 0 ]; then
  "$@"
else
  guard=(guard --name "$RUN_ID" --first-output "$FIRST" --stall "$STALL")
  [ "${OPS_GUARD_KILL:-0}" = 1 ] && guard+=(--kill)
  python3 "$OPS" "${guard[@]}" -- "$@"
fi
rc=$?

case "$rc" in
  0) state=done ;;
  130|137|143) state=cancelled ;;
  *) state=failed ;;
esac
if [ "$registered" = 1 ]; then
  ops update "$RUN_ID" "final_state=$state" "state_source=ops_job.sh exit $rc" \
    || echo "[ops] WARNING: final_state not written" >&2
fi
exit "$rc"
