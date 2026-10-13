---
name: pull-evals
description: >
  Use this skill when eval jobs have finished and their numbers need to be collected —
  "pull the eval results", "did the evals land", "check on the eval sweep", "what did
  <run> score". It cross-references the finished jobs against the run registry
  (`ops runs`), validates the numbers before anyone believes them, then records them with
  the record-results skill (`ops result add --check`) and marks each job done
  (`ops update`). Pairs with the `run-evals` skill, which launched them.
---

# Pulling eval results

The launch half is `run-evals`. This is the other half: finished jobs → validated rows in
the `ops` results log, with the run registry kept honest along the way.
`OPS=/accounts/projects/berkeleynlp/prasann/bin/ops`. results-hub is retired (2026-10-12);
nothing is ingested or pushed there any more.

**Nothing gets reported or recorded until it has been cross-referenced against the
registry and its generations have been looked at.** Both of those exist because numbers
from this harness have been wrong in ways that look exactly like results.

## Step 1: start from the registry, not from weka

The registry is the authority on what was launched:

```bash
$OPS runs --experiment E --json          # every registered run, folded; evals have kind=eval
$OPS status --experiment E               # live / finished state of each, all clusters
```

**Let it drive the whole pull.** The work list is every eval run whose `final_state` is not
yet set. Each carries a `job_id` (the Beaker experiment ID); query that ID directly rather
than globbing weka and inferring which job wrote which file. Guessing is how a base-pass
file gets mistaken for an xlong-pass file on the same checkpoint. Launches from before the
registry may only be in a YAML ledger under `records/eval_launches/` — use it as a
fallback source of the same facts (and register the jobs you pull from it).

```bash
beaker experiment results <job_id>      # per job, straight to its experiment
```

Then close the loop **in the other direction** too:

- **Every registered eval job without a `final_state`** → resolve it to a result JSON or
  an explicit reason it has none (still running, failed, preempted). Do not quietly
  record the subset that happens to be there and call the sweep done.
- **Every result JSON with no registered job** → an **unrecorded launch**. Do not record
  it from guesswork: its YaRN factor, decode knobs, eval bundle and realized `batch_size`
  cannot be recovered from the JSON, and they are what make the row comparable or not.
  Reconstruct them from the Beaker job's command and register it after the fact
  (`ops register ... --notes "registered after the fact"`), or leave the result out and
  say so.
- **A registered job with no `job_id`** → treat as unrecorded too. Backfill the ID from
  the Beaker workspace (`ops update RUN job_id=...`) before pulling it.

A sweep is `done` only when every job is `done` or explicitly `failed` with a reason.
Report the counts (`done / failed / still running`), not just the numbers you managed to
collect.

Result paths, for reading the JSON the experiment produced:

```
# flat copies, keyed by run + eval tag
/weka/oe-training-default/ai2-llm/checkpoints/prasanns/_eval_results/<RUN><_TAG>_<TASK>_multirung.json
# per-run copies
/weka/oe-training-default/ai2-llm/checkpoints/prasanns/<RUN>/eval<_TAG>/<TASK>_multirung.json
# generations alongside each
<same path>.generations.jsonl
```

The `_TAG` suffix is the `--eval-tag` from launch — this is how the base pass, the
native xlong pass and each YaRN group stay separate. Two passes of the same task on the
same checkpoint are different rows, never a duplicate to be deduped away.

(olmo-eval `launch_ctc_suite.py` evals write their own per-cell JSONs under the run's
weka results root; the same rules apply.)

## Step 2: a nonzero exit is not necessarily a lost result

Scoring finishes before the upload/cleanup steps, so a job can exit 1 with a perfectly
valid result JSON already written (a datalake 503 or an upload hiccup has done this).
**Check for the JSON before requeueing anything** — rerunning a completed eval burns
GPU hours and, on some harnesses, silently returns cached numbers.

Pull job state and artifacts by experiment ID with the Beaker commands in `beaker.md`
(repo root), which documents the invocations that actually work.

## Step 3: validate before believing

Do all of these before a single number is quoted or recorded.

1. **Read the generations.** `print_gen_sample.py` runs at the end of each job; its
   output is in the job log, and the `.generations.jsonl` is on weka. Uniform ~0.000
   across tasks and rungs is **never** a model result — it is a harness bug (wrong
   tokenizer family, truncated prompt tail, degenerate prefill).
2. **Check `parse_rate`.** f1 ≈ 0.000 at `parse_rate` 1.0 is the signature of a
   truncated prompt tail: the model parsed fine and answered a question it never saw.
3. **Check monotonicity across rungs.** A single rung 5–6× below *both* its neighbours
   is a config artifact, not a capability cliff. The known causes: a missing YaRN copy
   at ≥256k, an undersized `max_length`, an unchunked prefill.
4. **Check the ≥256k rungs carry a YaRN factor** in the job's registered `notes` /
   `ckpt_path` (a `_yarn2` copy, or olmo-eval's r256k default). A ≥256k number without
   one measures RoPE extrapolation off the end of the model and is unusable.
5. **Check `eval_size` per rung** in the JSON — the realized count, not `--max-test`.

If a check fails, the finding is "this job needs rerunning", not a result. Mark it
`ops update <run_id> final_state=failed notes="<reason>"` and say so in the report.

## Step 4: record — follow the `record-results` skill

One result row per **(task, rung)** — not one per job; a job covering four rungs yields
four rows. `ops result add --json` reads `*_multirung.json` directly (one row per
`<task>_<rung>` key). Files written since 2026-10-12 carry `_meta.eval_run_id` (the pass's registered
id), `_meta.eval_size`, the eval job id and commit, so `--json FILE --check` is enough. For an older
file, pass what it lacks:

```bash
$OPS result add --json <result JSON> --check \
  --run-id <the EVAL run_id of this pass (ops runs)> --train-run <the TRAINING run_id scored> \
  --experiment E --eval-size <realized eval_size> \
  --cot-label no-cot --eval-backend native --eval-job-id <beaker id> \
  --ckpt-path <the ckpt AS LAUNCHED, incl. _yarn2/_yarn4/_yarn8> --git-commit <EVAL commit>
```

Investigate **every** `--check` warning (jump vs prior rows, duplicate, eval_size<500,
missing provenance) before quoting the number — the full checklist is in `record-results`.

Rules carried over from the results-hub era, still binding:

- **One pass, one `run_id`; one model, one `train_run`.** Each eval pass records under its
  own eval `run_id`, and every pass over one checkpoint shares the same `train_run` —
  including its YaRN passes: `<step>_yarn2/4/8` are config-only copies of the *same
  weights* (`make_yarn_copy.py`), so their `train_run` is the trained run, not the copy
  (`ops` strips `_yarnK` from the ckpt path; pass `--train-run` if in doubt). The YaRN factor
  lives in `ckpt_path` (the copy launched), the eval run id's variant and its `notes`.
- **The realized settings, not the passed ones.** The runner forces bs=1 for
  landmark/compressive and for all xlong rungs, and raises `max_length` 10% over the rung;
  record what ran (in the eval job's `notes`), not what `eval_command` passed.
- **The eval bundle** is the standard `_eval_bundle_eval500_v2_clean` unless the launch
  notes record an override.

## Step 5: mark the jobs

For each job whose results were recorded:
`$OPS update <eval run_id> final_state=done` (or `$OPS status --record` to write every
finished state at once). Jobs that failed validation get
`$OPS update <eval run_id> final_state=failed notes="<reason>"`. Then update the
experiment's `STATUS.md` **Current best numbers** and run `$OPS exp index`
(`record-results` step 4). If a YAML ledger exists for these launches, flipping its jobs
to `done` is optional history — it stays gitignored and is never committed.

## Step 6: report

**Always end by reporting what was recorded.** One line per checkpoint — the `train_run`,
the tasks and rungs it covers, and the row count (`$OPS result show --experiment E`):

```
Recorded via ops result add:
  q35-landmark-256K-partialrope   contra/nq/outlier/rerank/oolong @ 2k-32k + 64k,128k   38 rows
                                  ...same train_run @ 256k,512k (yarn2) + 1M (yarn4)    15 rows
```

(The YaRN passes appear under the *same* train_run as their native rungs — one model, one
row group, a length curve that runs end to end — each with its own eval run_id.)

Then the numbers themselves:

- **Quote a resolution, not three decimals.** Binomial standard error is ±0.021 at
  f1≈0.70 and ±0.010 at f1≈0.95 for ~500 examples. Check any difference against that
  before calling it real — and remember run-to-run seed variation adds more on top.
- **Flag any sub-500 eval inline, next to the number**, with its size and error bar:
  `f1=0.83  ⚠ eval_size=100 only (±0.038)`. Never present a sub-500 number bare.
- Write `eval_size`, never `n` — in this project `n` means corpus size.
- State the sweep's completeness: how many jobs done, failed, still running.

## Checklist before reporting results

- [ ] Registry read first (`ops runs --experiment E`); work list = eval jobs without a
      `final_state`, fetched by `job_id`
- [ ] No result JSON recorded without a matching registered run (`ops runs`)
- [ ] No registered job left silently unaccounted for
- [ ] Generations actually read; `parse_rate` and monotonicity checked
- [ ] ≥256k rungs carry their YaRN factor
- [ ] One row per (task, rung) via `ops result add --json ... --check`; every warning investigated
- [ ] Each pass under its own eval `run_id`; the same `train_run` across the checkpoint's rows,
      including its YaRN passes
- [ ] Jobs marked `final_state=done`/`failed` (`ops update` / `ops status --record`)
- [ ] STATUS.md numbers updated + `ops exp index`
- [ ] Numbers reported at their resolution; sub-500 evals flagged inline
