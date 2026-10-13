# INCORPORATE: run registry + results log (agent A)

Post-refactor repo changes so every launch lands in `ops/state/runs.jsonl` and every number in
`ops/state/results.jsonl`. Paths below are **pre-refactor**; re-resolve each one against the new
`recipes/…` / `research/…` layout first (`git log --follow` or grep for the quoted symbol).
Invoke ops via the stable path `/accounts/projects/berkeleynlp/prasann/bin/ops` (or
`python3 <repo>/ops/bin/ops` inside Beaker, where `~/bin` does not exist). ops is stdlib-only,
so calling it from a launcher costs ~50 ms and needs no env.

## 1. Emit a registry row from the shared Beaker launcher

File: `src/scripts/train/memexpress/ctc_suite/beaker_ctc_suite.py` (post-refactor likely
`recipes/…/beaker_ctc_suite.py`). Insert right after the line
`print(f"[beaker_ctc_suite] SUBMITTED id={wl_id} name={wl_name}")` (~line 367):

```python
_ops = os.path.join(REPO_ROOT, "ops")             # REPO_ROOT = the launching checkout
sys.path.insert(0, _ops)
from opslib import registry as _reg              # stdlib-only
_reg.register(
    run_id=run_name, experiment=opts.experiment or f"ctc_suite_{opts.task}",
    kind="train", cluster="beaker", job_id=wl_id, task=opts.task, arm=opts.variant,
    ckpt_path=save_folder,
    wandb_url=f"https://wandb.ai/{WANDB_ENTITY}/{WANDB_PROJECT}/groups/{wandb_group}",
    command=" ".join(sys.argv), source="launch",
    expected_minutes=opts.expected_minutes,        # new optional CLI flag, default None
    **_reg.git_info(REPO_ROOT),                    # git_commit/git_branch/dirty
)
```
- Add `--experiment` (campaign key, e.g. `ds64`) and `--expected-minutes` argparse flags.
- Beaker ships the **pushed** commit, so if `dirty` is true print a warning (the shipped commit
  is `git_commit`, not the working tree). Do this in `dry_run` too? **No** — only after submit.
- Wrap in `try/except Exception as e: print(f"[ops] registry write failed: {e}")`; a registry
  failure must never fail a launch.

## 2. Shared sbatch wrapper

Add a 3-line prologue to the shared local launcher(s) (`*local*.sbatch` in
`src/scripts/train/memexpress/**`; ideally factor them into one `recipes/common/run_local.sbatch`):

```bash
OPS=/accounts/projects/berkeleynlp/prasann/bin/ops
python3 "$OPS" register --run-id "$RUN_NAME" --experiment "${OPS_EXPERIMENT:?set OPS_EXPERIMENT}" \
  --cluster slurm --job-id "$SLURM_JOB_ID" --node "$(hostname -s)" --kind "${OPS_KIND:-train}" \
  --log-path "$LOG" --ckpt-path "$SAVE_FOLDER" --git "$REPO" --command "$0 $*" || true
```
and an epilogue `python3 "$OPS" update "$RUN_NAME" final_state=$([ $rc = 0 ] && echo done || echo failed) state_source=sbatch-exit || true`.
Lambda (`run_ctc_lambda.sbatch`): same, `--cluster lambda`; lambda has no `/accounts` share, so
the registry row must be written from the Berkeley side at submit time by the rsync/ssh driver,
not inside the job. Remember lambda's `sbatch` FAILED-on-success trap: do **not** derive
`final_state` from the lambda exit code; leave it to `ops status`.

## 3. Native multirung evaluator -> contract results

- `src/scripts/ctc_eval/eval/eval_lc_native.py` writes `{model_path, <task>_<rung>: f1, _meta}`
  with **no eval_size**. Change: add per-key `"_eval_size": {"<task>_<rung>": N}` (N = examples
  actually graded after `--max-test-samples` and too-long skips) plus `"_meta": {"git_commit",
  "eval_job_id": $BEAKER_WORKLOAD_ID}`. Then `ops result add --json FILE` works unattended
  (`results.rows_from_json` already parses the multirung shape; extend it to read `_eval_size` /
  `_meta` — a 5-line change in `ops/opslib/results.py`).
- Launcher `debug/outlier_lengthmix_scaling/beaker_native_lengthmix_eval.py` (~line 88): after
  the eval command, append `&& python3 /gantry-runtime/ops/bin/ops result add --json /results/…_native_multirung.json --check`
  — but the container has no write path to the login-node state file, so instead: harvesters
  (`debug/ds64/harvest*.py`, `debug/flop_scaling/collect*.py`) call `results.add(...)` when they
  pull the JSON. Prefer the harvester route.

## 4. Converge with `results_io.py`

File: `src/scripts/train/memexpress/ctc_suite/ctc_suite/results_io.py`. Keep its §8 per-rung
JSON, but make `write_result()` also call `ops.results.add(...)` with this mapping (already
implemented in `results.rows_from_json`, so passing the §8 record through it is enough):

| results_io §8 | ops results.jsonl |
|---|---|
| `metric_name` / `metric_value` | `metric` / `value` |
| `rung_tokens` (int) | `rung` |
| `provenance.ckpt_path` | `ckpt_path`; `run_id` = basename minus `stepN` unless given |
| `provenance.git_commit` / `eval_backend` | same names |
| `eval_size`, `cot_label`, `task` | same |
| `small_eval_ok` / `small_eval_warning` | dropped; ops auto-adds flag `eval_size<500` + `se` |
| `model`, `arm`, `complexity_class`, `trained_ctx`, `aux*` | not carried (arm lives on the run row) |
| — | `eval_job_id` (new: add `provenance.eval_job_id` = `$BEAKER_WORKLOAD_ID`/`$SLURM_JOB_ID`) |

Then retire `results/ctc_suite/all_results.jsonl` (+ its `.lock`) as a sink; ops is the log.

## 5. Retire per-campaign ledger writers

Replace each `open(LEDGER, "a")` block with `registry.register(...)` (fields: run_id=run name,
experiment=<campaign>, task, arm, cluster, job_id=the parsed id, ckpt_path, git via `git_info`,
`command`). Delete the `LEDGER =` constant afterwards; the old TSVs stay as history (already
backfilled; `ops backfill ledgers` is idempotent if one more row slips in).

| file | line | note |
|---|---|---|
| `debug/flop_scaling/launch_grid35.py` | 184 | job_id column held literal `beaker35`; real id was only in notes |
| `debug/flop_scaling/launch_evals.py` | 62 | eval rows -> `kind="eval"`, `parent_run=<train run>` |
| `debug/ds64/launch_ds64.py` | 562 | headerless 10-col rows; notes captured gantry stderr instead of the id |
| `debug/ds64/launch_ns_evals.py` | 47, 51 | writes `ns_evals.tsv` (not `*LEDGER*`, so **not backfilled**) |
| `src/scripts/train/memexpress/cpt/ffndrop/launch_ffndrop_cpt.py` | 84 | |
| `src/scripts/train/memexpress/cpt/softdetach/launch_softdetach_cpt.py` | 92 | |
| `src/scripts/train/memexpress/cpt/softdetach/ctc_eval_ledger.py` | — | EVAL_LEDGER writer -> `kind="eval"` |
| orchestrators `debug/flop_scaling/orchestrate*.py`, `debug/ds64/orchestrator_ds64.py`, `…/ffndrop/pipeline.py` | state writes | on each terminal transition also call `registry.update(run, final_state=…, state_source=<state file>)` |

Also: `debug/flop_scaling/orchestrator35_state.json` in the working tree was overwritten by a
4-run orchestrator (HEAD has the 93-run grid). Give each orchestrator its own state filename.

## 6. Skills: drop results-hub

- `.claude/skills/run-evals/SKILL.md`: rule 4 ("launch ledger entry recording every results-hub
  column", ~line 21) and the "shared view … is its row in ../results-hub" section (~line 184)
  -> replace with `ops register --kind eval --parent-run <train run> …` at launch.
- `.claude/skills/pull-evals/SKILL.md`: replace the ingest step (`../results-hub` add/ingest,
  lines ~5–15, 30, 45–47, 86) with `ops result add --json <file> --eval-size N --check`; the
  ledger cross-reference becomes `ops runs --experiment E --json` joined on `run_id`.
- Memory/CLAUDE.md lines naming results-hub (`results-hub-central-table`, `results-csv-slicing`,
  `results-hub-eval-provenance`) -> point at `ops result show` (agent C owns memory edits).
- After this lands, results-hub's 415 Prasann rows live in results.jsonl with
  `source_json=results-hub:<id>`; amandab's 2195 rows were intentionally not imported.

## 7. Known data caveats from the backfill (for whoever quotes old numbers)

- 1545/1975 backfilled result rows have `eval_size=null` (ds64/flop_scaling CSVs never recorded
  it), so they carry no SE. All 1975 are `provenance_incomplete` (no eval_job_id or ckpt_path).
- `results/flop_scaling/results_scale_*.csv` re-list the 4B `prior-dense-*` rows under every
  scale (0.8b…27b) with rescaled FLOPs but the 4B f1 — backfill keeps one copy per run/task/rung.
