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

The launch half is `run-evals`. This is the other half: finished Beaker jobs → validated rows in
`../results-hub`, with the ledger kept honest along the way.

**Nothing is reported or ingested until it has been cross-referenced against the ledger and its
generations have been looked at.** Both exist because numbers from this harness have been wrong in
ways that look exactly like results.

## Step 1: start from the ledger, not from the results directory

```
records/eval_launches/<YYYY-MM-DD>_<run-name>[_<tag>].yaml
```

The work list is every job whose `status` is not `done`. Each carries a `beaker_experiment_id` —
query it directly rather than listing `ctc_eval/` and inferring which pass wrote which file
(filenames are `<task>_<rung>_<attn>[_<tag>].json`, so two bundles or two query-positions on one
checkpoint are told apart only by their `--tag` — the second pass is refused at startup rather
than allowed to overwrite, but the *name* still says nothing about which pass wrote it, so read
`bundle_root` and `_meta.query_position` inside the file).

```bash
beaker experiment get <experiment_id> --format json     # state: jobs[-1].status
beaker job logs <job_id> --tail 200                      # the per-rung summary table
```

Close the loop in both directions:

- **A ledger job still `submitted`** → resolve it to results or to an explicit reason (running,
  failed, preempted). Do not ingest the subset that happens to be there and call the sweep done.
- **A result file with no ledger job** → an unrecorded launch. Its model-side columns (model_type,
  attention_type, pipeline_stage, slug) are not in the JSON and cannot be recovered from it.
  Reconstruct them from the Beaker job's command and backfill a ledger entry, or leave it out and
  say so.
- **A ledger job with no `beaker_experiment_id`** → same treatment; backfill the ID first.

Report counts (`n done / n failed / n running`), not just the numbers you collected.

## Step 2: a nonzero exit is not a lost sweep

The launcher passes `--keep-going`, so a failed rung is recorded and the sweep continues — and the
process then exits 1 even though every other rung wrote a valid result. **Check what landed before
requeueing anything.** The log's closing block lists each rung's summary line and, separately, the
rungs that FAILED.

## Step 3: read the JSONs (weka is not mounted on this host)

Results live on the weka mount, which this machine does not have. Read them from a node:

```bash
# what landed
python debug/fast_bundle/probe_weka.py 'ls -la /weka/.../checkpoints/prasanns/<RUN>/ctc_eval'

# one rung, minus the generations blob, plus the first few generations
python debug/fast_bundle/probe_weka.py 'python -c "
import json; d=json.load(open(\"/weka/.../<RUN>/ctc_eval/contradiction_8k_chunked.json\"))
g=d.pop(\"generations\",[]); print(json.dumps(d,indent=2)); print(json.dumps(g[:5],indent=2))"'
```

`probe_weka.py` submits a GPU-less job that runs one read-only command; the output comes back in
`beaker job logs`. Save each JSON you are ingesting to a local scratch copy under `debug/` (not
committed) so the ingest command has a file to read.

Each result file carries: `ladder`, `spec`, `rung`, `attn`, `backend`, `eval_size`, `metrics`,
`standard_errors`, `primary_metric`, `primary_value`, `parse_rate`, `truncated`, `prompt_tokens`,
`warnings`, `bundle` / `bundle_root` / `bundle_kind`, `share_prefix`, `generations`, `provenance`
(`ckpt`, `data_path`, `git_commit`, `query_position`, `max_length`, `max_new_tokens`) and `_meta`.

## Step 4: validate before believing

All of these, before a single number is quoted:

1. **`warnings`** — the file says what went wrong. `format fingerprint check DISABLED` means
   train/eval format compatibility is UNVERIFIED; a truncation warning means prompts exceeded
   `max_length`; a small-eval warning means the floor was not met.
2. **`parse_rate`** — below 1.0 means generations did not parse. A parse failure and a wrong answer
   both score zero, so this is a decoding/truncation problem, not a capability one.
3. **Read the generations.** Uniform ~0.000 across tasks and rungs is never a model result — it is
   a wrong tokenizer family, a truncated prompt tail, or a degenerate prefill.
4. **`truncated` and `prompt_tokens`** — a rung label bounds corpus size, not prompt length. A
   nonzero `truncated` means examples were counted as over-long rather than graded.
5. **`eval_size`** — the realized count. scifact is 299 and fast-bundle oolong is 100; both are
   below the 500 floor.
6. **Monotonicity across rungs** — one rung far below both its neighbours is usually a config
   artifact, not a capability cliff.
7. **`bundle` / `bundle_kind`** — a `fast` number is a different measurement from a reliable one
   (outlier +0.215/+0.261, contradiction −0.102…−0.175). Never mix them into one series. Fast
   outlier at 8k has a measured 0.203 shortcut; don't quote it.

A failed check is "this job needs rerunning", not a result. Mark it `failed` in the ledger with the
reason and say so in the report.

## Step 5: ingest into results-hub

One row per **(task, rung)** — a single job covering many rungs produces many rows.

**`eval_version` is never typed.** Every run writes `_meta: {ladder_version, eval_bundle,
query_position}`, and `ingest_eval_json.py`'s `meta_from_meta_block` turns that into `eval_version`
(`v1` / `v2` / `fast`) and `eval_set_weka_pointer`. Precedence is args-derived < provenance < `_meta`
< CLI flags, but a `--eval-version` that contradicts `_meta` is a **hard error by design** — so do
not pass it. Copy the value only for a legacy file that predates `_meta`. The old instruction "type
`v2`, the only supported ladder" is what this replaced: it silently relabelled shared-corpus numbers
as reliable ones.

⚠ **Known gap: `ingest_eval_json.py` does not yet recognise the per-rung `ctc-eval` shape.** Its
shape-2 branch treats every top-level dict of numbers as a task, so one result file yields three
junk rows named `metrics`, `standard_errors` and `prompt_tokens` (verified). Until it learns this
shape, ingest with `add_result.py` and copy the two `_meta` fields across by hand:

```bash
cd ../results-hub
python add_result.py \
    --eval-name contradiction --eval-context-length 8192 \
    --eval-data-quantity 500 --metric-name f1 --metric-value 0.83 \
    --extra-metrics '{"precision":0.9,"recall":0.8}' \
    --eval-version v2 \
    --eval-set-weka-pointer /weka/.../_eval_bundle_eval500_v2_clean \
    --git-commit <provenance.git_commit> \
    --model-type compressive --model-subtype block64 \
    --attention-type doc_chunked --pipeline-stage SFT --chat-template chat \
    --weka-model-location <provenance.ckpt> \
    --model-train-data-hparams q4b-compressive-block64-32K \
    --decoding-hparams-other "backend=native max_length=40960 max_new_tokens=64 query_position=both share_prefix=false" \
    --notes "<beaker experiment id>; <training description>; bundle=v2_clean kind=reliable"
```

Both `--eval-version` and `--eval-set-weka-pointer` here are **transcribed from the file's `_meta`**,
not chosen. `--git-commit` comes from `provenance.git_commit` (the commit the node actually ran) —
prefer it over `--code-repo`, which records this machine's HEAD instead. `who_ran`, `date_eval_ran`
and `id` auto-fill.

Model-side columns come from the ledger, never re-derived: `model_type`, `model_subtype`,
`attention_type` (`--attn chunked` → `doc_chunked`), `pipeline_stage`, `chat_template`,
`weka_model_location`, `model_train_data_hparams`, `other_notes`. The landmark top-k columns have no
flag in this harness — leave them blank.

### `model_train_data_hparams` is a short model slug

```
q35-landmark-256K-partialrope    q35-dense-256K    q4b-compressive-block64-32K
```

The Compare tab pivots methods by this field, so prose turns it into soup. Derive it from columns
you already have — family (`q35`, `q4b`) + method (`dense`, `landmark`, `compressive-block64`) +
trained context (`32K`, `256K`) + whatever distinguishes it from its siblings (`partialrope`,
`cptmix`). Lowercase kebab-case, uppercase `K`/`M`.

- **It describes the model, not the eval.** Bundle, query-position, backend and batch size are
  eval-side and belong in `decoding_hparams_other` / `other_notes`.
- **Identical for every row of one checkpoint** — all tasks, all rungs, all passes, including a
  `fast`-bundle pass. Different slugs split one model into two methods in Compare; `eval_version`
  is what keeps the fast series apart, and it already does that on its own.
- The long training description goes in `other_notes`, where nothing is lost.
- **If the right slug is unclear, ask.** A wrong slug either merges two methods or splits one, and
  nothing downstream flags it. Once derived, write it back to the ledger's `checkpoint.model_slug`
  and reuse it verbatim.

Then `git add results.csv && git commit && git push`. `results.csv` is union-merged, so concurrent
pushes of different rows don't conflict.

## Step 6: mark the ledger

For each job whose results were ingested, update its entry in place:

```yaml
  - name: main-2k-32k
    beaker_experiment_id: 01KZQ58X21CM4GPW9R70YK0W5D
    status: done                      # was: submitted
    rungs_ingested: [contradiction@2k, contradiction@8k, nq@3k, ...]
    pulled_at: 2026-08-12
    notes: ""
```

Jobs failing validation get `status: failed` and a reason. Update the
`records/eval_launches/README.md` row too. Do this in the same session as the push so the two never
drift — but **do not commit the ledger**; `results.csv` is the only thing committed here.

## Step 7: report

One line per checkpoint pushed, slug first, so the slug choice is visible before it propagates:

```
Pushed to results-hub:
  q4b-compressive-block64-32K   contra/nq/outlier/rerank/oolong @ 2k-32k   18 rows   [v2]
                                contradiction @ 64k-1M                      6 rows   [fast — separate series]
```

Then the numbers:

- **Quote a resolution, not three decimals.** The file carries `standard_errors`; use it. ±0.021 at
  f1≈0.70 and ±0.010 at f1≈0.95 for ~500 examples, and seed variation adds more on top.
- **Flag any sub-500 eval inline**, with its size and error bar: `f1=0.83  ⚠ eval_size=299 (±0.027)`.
- Write `eval_size`, never `n` — in this project `n` means corpus size.
- Say which bundle produced each number whenever a fast row is in the report.
- State completeness: how many jobs done, failed, still running.

## Checklist before reporting results

- [ ] Ledger read first; work list fetched by experiment ID
- [ ] No result ingested without a matching ledger job; no ledger job left unaccounted for
- [ ] `warnings`, `parse_rate`, `truncated`, `eval_size` checked and generations actually read
- [ ] `eval_version` came from `_meta`, never typed from memory; fast rows kept as their own series
- [ ] One row per (task, rung); model-side columns taken from the ledger
- [ ] `model_train_data_hparams` a short slug, identical across the checkpoint's rows (asked if
      unclear); long training description in `other_notes`
- [ ] `results.csv` committed and pushed — the only thing committed here
- [ ] Ledger jobs flipped to `done`/`failed`, saved locally, NOT committed
- [ ] Numbers reported at their resolution; sub-500 evals flagged inline
