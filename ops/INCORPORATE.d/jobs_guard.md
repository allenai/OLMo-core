# jobs / stall / guard / hooks: repo-side follow-ups (agent B)

`ops status`, `ops stall`, `ops guard` and the two PreToolUse hooks work without any repo change.
They get much better once launchers tell the registry what to expect. Paths below are pre-refactor
(branch `prasann/landmark`), so remap them if the refactor moved them.

## 1. Launchers register runs (log_path + expected_minutes)

`ops status` flags OVERRUN only when `expected_minutes` is known. It flags STALE-LOG for Beaker runs
and for logs other than slurm `--output` only when `log_path` is known.

- `src/scripts/train/memexpress/ctc_suite/run_ctc_local.sbatch`, `run_ctc_lambda.sbatch`,
  `ctc_suite/run_rung_eval.sbatch`: right after the `#SBATCH` header add
  `/accounts/projects/berkeleynlp/prasann/bin/ops register --run-id "$RUN" --experiment "${OPS_EXPERIMENT:-ctc_suite}" --cluster slurm --kind train --job-id "$SLURM_JOB_ID" --node "$(hostname -s)" --log-path "/data/prasann/joblogs/ctc_suite_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" --ckpt-path "$SAVE_LOCAL" ${EXPECTED_MINUTES:+--expected-minutes "$EXPECTED_MINUTES"} || true`
  (Lambda: `--cluster lambda`; the ops checkout is not visible there, so register from the
  login-side submit wrapper instead).
- `src/scripts/train/memexpress/ctc_suite/beaker_ctc_suite.py` (`main()`) and every
  `debug/*/launch*.py` / `fire_evals.sh` that calls `gantry run`: after the experiment id is
  parsed, call `opslib.registry.register(run_id=..., experiment=..., cluster="beaker", job_id=<EXP-id>, expected_minutes=..., git_commit=<pushed sha>, command=" ".join(sys.argv))`
  (or shell out to `ops register`). Today the experiment ids only reach per-family
  `LAUNCH_LEDGER.tsv` files, which agent A backfills.
- Each launcher should declare `EXPECTED_MINUTES` (or derive it from steps x measured step time)
  so OVERRUN (> 2x expected) fires.

## 2. Wrap sbatch payloads in `ops guard`

In the same sbatch files, prefix the payload, for example:

```bash
OPS=/accounts/projects/berkeleynlp/prasann/bin/ops
python3 $OPS guard --name "$RUN" --first-output 600 --stall 1200 -- \
  "$PY" -m torch.distributed.run --nproc_per_node="$NGPU" ...
```

`ops` itself is stdlib-only and runs fine from NFS. Only the payload interpreter must be on
node-local storage. The STALL block is written into the slurm log, so `ops stall` or a human reading
the log sees the diagnosis without attaching to the job. Data builds (`src/scripts/data/**/*.sbatch`,
`debug/ctc_contra_iid_eval/build_*.sbatch`) should use the defaults (`--first-output 120 --stall 300`).

## 3. Bare `torchrun` → `$PY -m torch.distributed.run`

37 tracked `*.sbatch` / `*.sh` files call bare `torchrun`:
`git --no-optional-locks grep -ln "^\s*torchrun \|[;&|] *torchrun " -- '*.sbatch' '*.sh'`.
Most of them export `PATH="$ENV/bin:$PATH"` first, and that path is safe only while `$ENV` points at
`/data`. Several launchers fall back to `/scratch/...` when the `/data` env is missing, and in that
case the fallback lands on the NFS interpreter without any warning. Change:
- `torchrun ...` → `"$PY" -m torch.distributed.run ...` with `PY=$ENV/bin/python`.
- Remove the `/scratch` fallback, or make it exit loudly:
  `case "$PY" in /data/*) ;; *) echo "FATAL: not node-local: $PY"; exit 1;; esac`.

## 4. slurm `--output` off NFS

37 tracked sbatch files have `#SBATCH --output=/accounts/...` or `/scratch/...`. List them with
`git --no-optional-locks grep -ln "^#SBATCH --output=/accounts\|^#SBATCH --output=/scratch" -- '*.sbatch'`.
Examples: `src/scripts/train/memexpress/attn_explore/run_q06b_*_local_mooney.sbatch`,
`debug/ctc_llama/*.sbatch`, `src/scripts/data/ctc_suite/convert_*.sbatch`. Move each one to
`/data/prasann/joblogs/%x_%j.log`, which is the known step-0 deadlock fix in
`memory/local-train-nfs-log-hang.md`. `ops status` flags live jobs with `LOG-ON-NFS`, and the
bash hook advises on them at `sbatch` time.

## 5. Stack dumps for hung jobs

- py-spy is installed nowhere on the cluster (checked PATH, `/data/prasann/conda/envs/*/bin`, and
  `/scratch/users/prasann/conda/envs/*/bin`). Install it once per node-local env:
  `$PY -m pip install py-spy`. Both `ops stall` and `ops guard` pick it up from
  `/data/prasann/conda/envs/*/bin/py-spy` and print `py-spy dump` output for the busiest python
  process.
- In `src/scripts/train/memexpress/ctc_suite/train_ctc_suite.py` (and the other `train_*.py`
  entry points), at startup add:
  `import faulthandler, signal; faulthandler.register(signal.SIGUSR1, all_threads=True)`.
  `kill -USR1 <pid>` then dumps every Python stack to stderr without killing the rank. This works
  even without py-spy. Do not use `faulthandler.dump_traceback_later(N, exit=True)` as a watchdog
  (see `memory/faulthandler-300s-watchdog-trap.md`). Use `repeat=True, exit=False` only, if anything.

## 6. Early progress logging

Data builders and eval drivers should print at items 1, 2, 5, 10, 25, 50 and then every N, with
elapsed time, rate and ETA, and `flush=True` (`memory/log-progress-early.md`). Without that,
`ops guard` fires `--first-output` on legitimate slow starts, and `ops stall` cannot tell a livelock
from slow work. Start with `src/corpus_reasoning/data/*` generators,
`src/scripts/data/convert_longctx_tasks_to_sft.py`, and
`src/scripts/train/memexpress/ctc_suite/ctc_suite/run_rung_eval.py`.

## 7. Docs

- `CLAUDE.md` → "Local (Berkeley) cluster": add one paragraph covering `ops status` (one cached
  snapshot, so never poll squeue/beaker in loops), `ops stall <job|run>` (verdict first: NFS stall,
  GPU idle, hung, no output N min), and `ops guard -- CMD` for long local commands. Mention that the
  PreToolUse hooks deny foreground poll loops, login-node test runs, NFS interpreters in jobs,
  `/net` job I/O and bare torchrun.
- `beaker.md` "monitoring" section: point at `ops status` / `ops stall <EXP-id>` instead of
  hand-rolled `beaker experiment get` loops. The state mapping matches
  `debug/ctc_vllm_validation/beaker/job_status.py` plus CANCELLED/PREEMPTED.
- `local_cluster.md` trap table: add the row "job looks hung" → `ops stall <id>`.

## 8. Hook install (main agent)

Follow `ops/hooks/README.md`: symlink `bin/ops-hook-bash` and `bin/ops-hook-agent`, then add the
two PreToolUse entries to `~/.claude/settings.json`. The `claude-work` profile (`~/.claude-work`)
needs the same entries.
