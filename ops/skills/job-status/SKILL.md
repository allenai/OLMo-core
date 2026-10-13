---
name: job-status
description: >
  Use this skill WHENEVER checking on jobs or runs — "status?", "is it running", "did X finish",
  "how are the runs doing", "check on the sweep / the evals / training", "is anything stuck",
  "why is it slow", before reporting progress, and before launching something that might
  duplicate a live run. One `ops status` call covers Beaker, Berkeley slurm and Lambda at once;
  `ops stall ID` diagnoses one suspicious job. Replaces ad-hoc squeue/beaker/ssh polling and
  sleep loops.
---

# Job status

`OPS=/accounts/projects/berkeleynlp/prasann/bin/ops` (stdlib python, safe on the login node, a few seconds).

1. **One call:** `$OPS status` (registered runs that are live or finished recently, all clusters).
   Narrow with `--experiment E`; `--all` includes old finished rows; `--json` for parsing;
   `--fresh` bypasses the cache; `--no-beaker` / `--no-lambda` if one backend is down.
2. **Anything flagged** (STALL, OVERRUN = over 2x its `--expected-minutes`, NO-OUTPUT, FAILED,
   unknown) **or suspiciously slow:** `$OPS stall <job_id|run_id>`. It reads the log tail, process
   state (`D` + `nfs_wait_bit_killable` + ~0% CPU = NFS interpreter/log trap, not a hang or a
   compile), GPU use and last progress line. Act on its diagnosis; don't re-derive it by hand.
3. **A job that is not in `ops status`** was launched without `ops register`. Register it now
   (`$OPS register --run-id ... --experiment ... --cluster ... --job-id ...`) so the next check sees it.
4. **Report** per run: state, step or progress and ETA, and anything flagged. Quote numbers only
   after the `record-results` skill.

## Rules
- **Never write `sleep`/poll loops** (`while true; do squeue; sleep 60; done`, repeated `beaker
  experiment get`). To wait for a specific condition (a file appears, a log line matches, a job
  leaves the queue), use the **Monitor** tool with an until-condition, then check once with
  `ops status`. Otherwise just check again when the user asks.
- **Lambda: `sbatch`/`sacct` FAILED ≠ failed.** Lambda routinely reports FAILED for runs that fully
  succeeded. Verify by the loss curve plus the final checkpoint (see `lambda_cluster.md`).
- **Beaker:** status comes from the job JSON (`jobs[-1].status`); the only cancel that works is
  `beaker job cancel <job-id>`.
- After a run finishes: `$OPS update RUN_ID final_state=done|failed|cancelled` (or
  `$OPS status --record`), then the `record-results` skill for any eval output.
