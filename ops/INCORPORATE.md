# Incorporating `ops/` after the refactor lands

`ops/` was built on branch `prasann/ops-tools` (worktree `.claude/worktrees/ops_tools`), cut from
upstream `59a339f53`, the same base as `prasann/flopdrop_clean`. It only *adds* the `ops/` directory, so
merging it is conflict-free. Everything that has to touch existing repo files was deferred to this
checklist. The per-topic detail (file, line, exact change) lives in `INCORPORATE.d/`.

Run the steps in order. Each one is independent enough to be its own commit.

## Step 0: Merge and repoint (about 10 min)
1. On the refactored branch: `git merge prasann/ops-tools` (or `git checkout prasann/ops-tools -- ops/`).
2. Copy the *live* state from the worktree first, since the registry and results kept growing after the
   branch commit: `cp .claude/worktrees/ops_tools/ops/state/*.jsonl ops/state/`, plus
   `ops/experiments/` if any STATUS.md changed. Then `ops exp check`.
3. Repoint the stable symlinks to the merged checkout (no skill or memory text has to change):
   `ln -sfn <repo>/ops/bin/ops ~/bin/ops`, the same for `ops-hook-bash` / `ops-hook-agent`, and every
   `~/.claude/skills/{job-status,launch-run,record-results,close-out-experiment}`.
4. Memory: replace the absolute worktree path to `ops/experiments/INDEX.md` in `MEMORY.md` with the merged path
   (`INCORPORATE.d/knowledge.md` §0).
5. `git worktree remove .claude/worktrees/ops_tools` once nothing points there (`grep -r worktrees/ops_tools ~/.claude`).

## Step 1: Launchers write the registry (the biggest payoff)
See `INCORPORATE.d/registry_results.md` §1–2 and `INCORPORATE.d/jobs_guard.md` §1–2.
- The shared Beaker launcher (`beaker_ctc_suite.py`, or its new home under `recipes/`) calls
  `registry.register(...)` after the `SUBMITTED id=` line, with run_id, experiment, commit, wandb URL,
  ckpt_path and expected_minutes.
- A shared sbatch wrapper registers at start, writes `final_state` from the exit code at the end, and runs
  its payload under `ops guard`.
- Retire the 7 per-campaign ledger writers (`registry_results.md` §5). Keep the old TSVs as history; they
  are already backfilled.

## Step 2: Results path (results-hub retired)
`INCORPORATE.d/registry_results.md` §3–4 and §6:
- The native multirung evaluator emits contract rows (eval_size, eval job id, commit).
- Converge `results_io.py` with `ops/opslib/results.py`.
- Strip results-hub ingest from the `run-evals` / `pull-evals` skills and point them at `record-results`.

## Step 3: Knowledge layer in the repo
`INCORPORATE.d/knowledge.md` §1–4:
- Move `ops/skills/*` into `.claude/skills/` (or keep them in `ops/skills` and symlink).
- Fix `training-smoke-test`: priority is always `urgent`, and add the local/Lambda rules.
- CLAUDE.md: add a short "Research ops" section (ops CLI, experiments index, current CTC eval path), and
  move the upstream-generic text into a linked doc.
- records/: add the header template, regenerate the README with `ops exp records --write`, and apply the
  FEVER §4b fix.
- Update `launcher:` / `results:` paths in STATUS.md after the debug/ → research/ move.

## Step 4: Mechanical trap guards
`INCORPORATE.d/jobs_guard.md` §3–6:
- Bare `torchrun` → `$PY -m torch.distributed.run` (37 files).
- slurm `--output` off NFS (37 sbatch files).
- faulthandler SIGUSR1 stack dumps plus a py-spy install in the training env.
- Progress logging at items 1, 2, 5, 10.

Also add asserts for:
- `--max-tokens` below one step
- a non-empty save folder on a fresh (non-resume) run
- an unrepaired marker base at train start

## Already live (no incorporation needed)
- MEMORY.md slimmed to about 8 KB (backup in `memory/archive/`), 18 memory files marked `superseded_by`
  STATUS pages, and the audit's contradictions fixed.
- Stale `~/.claude/skills/.{run,pull}-evals.pre-repo-backup` moved to `~/.claude/skills-archive/`.
- `ops` CLI at `/accounts/projects/berkeleynlp/prasann/bin/ops`, with backfilled registry and results.
- Hooks and the 4 new skills: installed by Prasann (commands in `hooks/README.md` and the hand-off message).
