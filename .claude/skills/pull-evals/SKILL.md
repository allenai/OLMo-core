---
name: pull-evals
description: >
  Use this skill when eval jobs have finished and their numbers need to be collected —
  "pull the eval results", "did the evals land", "ingest into results-hub", "check on
  the eval sweep", "add these numbers to the hub", "what did <run> score". It
  cross-references the finished jobs against the launch ledger, validates the numbers
  before anyone believes them, ingests them into ../results-hub, and marks each job
  done in the ledger. Pairs with the `run-evals` skill, which launched them.
---

# Pulling eval results

The launch half is `run-evals`. This is the other half: jobs → validated rows in
`../results-hub`, with the ledger kept honest along the way.

**Nothing gets reported or ingested until it has been cross-referenced against the
ledger and its generations have been looked at.** Both of those exist because numbers
from this harness have been wrong in ways that look exactly like results.

## Step 1: start from the ledger, not from weka

The ledger is the authority on what was launched:

```
records/eval_launches/<YYYY-MM-DD>_<run-name>[_<tag>].yaml
```

**Read it first and let it drive the whole pull.** The work list is every job whose
`status` is not yet `done` — those are exactly the ones not yet in results-hub. Each
carries a `beaker_experiment_id`; query that ID directly rather than globbing weka and
inferring which job wrote which file. Guessing is how a base-pass file gets mistaken for
an xlong-pass file on the same checkpoint.

```bash
# the work list: ledger jobs not yet pulled
grep -B6 'status: submitted' records/eval_launches/<ledger>.yaml
# then, per job, straight to its experiment
beaker experiment results <beaker_experiment_id>
```

Then close the loop **in the other direction** too:

- **Every ledger job with `status: submitted`** → resolve it to a result JSON or an
  explicit reason it has none (still running, failed, preempted). Do not quietly ingest
  the subset that happens to be there and call the sweep done.
- **Every result JSON with no ledger job** → an **unrecorded launch**. Do not ingest it
  from guesswork: its YaRN factor, decode knobs, eval bundle and realized `batch_size`
  cannot be recovered from the JSON, and they are what make the row comparable or not.
  Reconstruct them from the Beaker job's command and backfill a ledger entry, or leave
  the result out and say so.
- **A ledger job with no `beaker_experiment_id`** → treat as unrecorded too. Backfill
  the ID from the Beaker workspace before pulling it.

A sweep is `done` only when every ledger job is `done` or explicitly `failed` with a
reason. Report the counts (`n done / n failed / n still running`), not just the numbers
you managed to collect.

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

## Step 2: a nonzero exit is not necessarily a lost result

Scoring finishes before the upload/cleanup steps, so a job can exit 1 with a perfectly
valid result JSON already written (a datalake 503 or an upload hiccup has done this).
**Check for the JSON before requeueing anything** — rerunning a completed eval burns
GPU hours and, on some harnesses, silently returns cached numbers.

Pull job state and artifacts by experiment ID with the Beaker commands in `beaker.md`
(repo root), which documents the invocations that actually work.

## Step 3: validate before believing

Do all of these before a single number is quoted or ingested.

1. **Read the generations.** `print_gen_sample.py` runs at the end of each job; its
   output is in the job log, and the `.generations.jsonl` is on weka. Uniform ~0.000
   across tasks and rungs is **never** a model result — it is a harness bug (wrong
   tokenizer family, truncated prompt tail, degenerate prefill).
2. **Check `parse_rate`.** f1 ≈ 0.000 at `parse_rate` 1.0 is the signature of a
   truncated prompt tail: the model parsed fine and answered a question it never saw.
3. **Check monotonicity across rungs.** A single rung 5–6× below *both* its neighbours
   is a config artifact, not a capability cliff. The known causes: a missing YaRN copy
   at ≥256k, an undersized `max_length`, an unchunked prefill.
4. **Check the ≥256k rungs carry a YaRN factor** in the ledger. A ≥256k number without
   one measures RoPE extrapolation off the end of the model and is unusable.
5. **Check `eval_size` per rung** in the JSON — the realized count, not `--max-test`.

If a check fails, the finding is "this job needs rerunning", not a result. Mark it
`failed` in the ledger with the reason and say so in the report.

## Step 4: ingest into results-hub

One results-hub row per **(task, rung)** — not one per job. A single job covering four
rungs produces four rows.

```bash
cd ../results-hub
python ingest_eval_json.py <result JSON> \
    --eval-context-length <rung> --eval-data-quantity <realized eval_size> \
    --model-type ... --attention-type ... --pipeline-stage ... \
    --weka-model-location <the ckpt AS LAUNCHED, incl. _yarn2/_yarn4/_yarn8> \
    --code-repo /path/to/OLMo-core     # records the EVAL commit, not the hub's
```

Take every non-metric column from the ledger entry rather than re-deriving it. The
fields most often gotten wrong:

- `weka_model_location` — the YaRN copy path when one was used, not the plain step dir.
- `decoding_hparams_other` — the **realized** `max_length` (after the harness's 10%
  raise) and the **actual** batch size. The runner forces bs=1 for
  landmark/compressive and for all xlong rungs, so what is in `eval_command` is what was
  *passed*, not what ran.
- `eval_set_weka_pointer` — the standard `_eval_bundle_eval500_v2_clean` bundle unless
  the ledger records an override.
- `other_notes` — Beaker job id, YaRN factor, the full training description, and
  anything making the row non-comparable.
- `who_ran` / `git_commit` / `id` auto-fill; don't hardcode them.

### `model_train_data_hparams` is a short model slug

This column is **a short, readable description of the model**, not a paragraph:

```
q35-landmark-256K-partialrope
q35-dense-256K
q4b-compressive-block64-32K
q4b-mcl-4lm-mean-block64-32K
```

The results-hub Compare tab pivots methods by this field, so it works as a slug and
turns into unusable soup as prose. Put the long training description — data mix, steps,
GBS, lr, parallelism, base init — in `other_notes` instead, where nothing is lost.

**Derive the slug yourself from the columns you already have**, every push:

| part | source | examples |
|---|---|---|
| family | checkpoint path / run name | `q35` (Qwen3.5-4B), `q4b` (Qwen3-4B) |
| method | `model_type` + `model_subtype` + `attention_type` | `dense`, `landmark`, `compressive-block64`, `mcl-4lm-mean` |
| context | the context the model was **trained** at | `32K`, `256K` |
| variant | whatever distinguishes it from its siblings | `partialrope`, `gate-temp`, `nocpt`, `cptmix` |

Rules that keep the field usable:

- Lowercase kebab-case; context in uppercase `K`/`M`.
- **Describes the model, not the eval.** The serving YaRN factor, top-k, and batch size
  are eval-side — they belong in `decoding_hparams_other`, never in the slug.
- **Identical for every row of one checkpoint** — all tasks, all rungs, all passes.
  Different slugs for the same model silently split it into two methods in Compare.
- **The YaRN copies are the same model, so they get the same slug.** `<step>`,
  `<step>_yarn2`, `<step>_yarn4` and `<step>_yarn8` are one checkpoint: `make_yarn_copy.py`
  writes a config-only copy that symlinks the *same weights*, so the only difference is
  how positions are served at eval time. Never write `-yarn2` into the slug and never
  give a scaled pass its own slug — that would turn one model's length curve into four
  unconnected methods, exactly hiding the comparison the xlong rungs exist to make.
  The distinction is preserved where it belongs: `weka_model_location` records the copy
  actually launched, and `decoding_hparams_other` / `other_notes` carry the factor.
- **Distinct whenever the model differs.** If two checkpoints would collide, add the
  variant that separates them rather than letting them merge.
- Once derived, write it back to the ledger's `checkpoint.model_slug` and reuse it
  verbatim for later passes of the same checkpoint.

**If the right slug is unclear — an unfamiliar architecture, two plausible variant
names, or a run name that doesn't say what makes it different — ask** rather than
inventing one. A wrong slug is worse than a slow one: it either merges two methods or
splits one, and nothing downstream flags it.

### Push

Then `git add results.csv && git commit && git push`. `results.csv` is union-merged, so
concurrent pushes of different rows don't conflict.

## Step 5: mark the ledger

For each job whose results were ingested, update its entry in place:

```yaml
- task: contra
  beaker_experiment_id: 01KZQ58X21CM4GPW9R70YK0W5D
  status: done                       # was: submitted
  result_json: /weka/.../_eval_results/<RUN>_base_contra_multirung.json
  model_slug_used: q35-landmark-256K-partialrope   # what went into model_train_data_hparams
  rungs_ingested: [2k, 8k, 16k, 32k]
  pulled_at: 2026-08-12
  notes: ""
```

Jobs that failed validation get `status: failed` and a `notes` reason. Update the
`records/eval_launches/README.md` row for the checkpoint too.

Update the ledger in the same working session as the results-hub push, so the two never
drift — but **do not commit the ledger**. It is local bookkeeping and
`records/eval_launches/` is gitignored; the only thing that gets committed and pushed
here is `results.csv`.

## Step 6: report

**Always end by reporting what was pushed, with the slug for each.** One line per
checkpoint pushed — the `model_train_data_hparams` slug, the tasks and rungs it covers,
and the row count — so the slug choice is visible and correctable before it propagates:

```
Pushed to results-hub:
  q35-landmark-256K-partialrope   contra/nq/outlier/rerank/oolong @ 2k-32k + 64k,128k   38 rows
                                  ...same slug @ 256k,512k (yarn2) + 1M (yarn4)         15 rows
  q35-dense-256K                  contra/nq/outlier/rerank/oolong @ 2k-32k              18 rows
```

(The YaRN passes appear under the *same* slug as their native rungs — one model, one
row group, a length curve that runs end to end.)

Then the numbers themselves:

- **Quote a resolution, not three decimals.** Binomial standard error is ±0.021 at
  f1≈0.70 and ±0.010 at f1≈0.95 for ~500 examples. Check any difference against that
  before calling it real — and remember run-to-run seed variation adds more on top.
- **Flag any sub-500 eval inline, next to the number**, with its size and error bar:
  `f1=0.83  ⚠ eval_size=100 only (±0.038)`. Never present a sub-500 number bare.
- Write `eval_size`, never `n` — in this project `n` means corpus size.
- State the sweep's completeness: how many jobs done, failed, still running.

## Checklist before reporting results

- [ ] Ledger read first; work list = jobs not yet `done`, fetched by experiment ID
- [ ] No result JSON ingested without a matching ledger job
- [ ] No ledger job left silently unaccounted for
- [ ] Generations actually read; `parse_rate` and monotonicity checked
- [ ] ≥256k rungs carry their YaRN factor
- [ ] One row per (task, rung); non-metric columns taken from the ledger
- [ ] `model_train_data_hparams` is a short slug, identical across the checkpoint's rows
      — including its YaRN passes (asked if unclear); long training description moved to
      `other_notes`
- [ ] `results.csv` committed and pushed — the only thing that gets committed here
- [ ] Ledger jobs flipped to `done`/`failed` (saved locally, NOT committed)
- [ ] Numbers reported at their resolution; sub-500 evals flagged inline
