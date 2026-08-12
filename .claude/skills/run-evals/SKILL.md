---
name: run-evals
description: >
  Use this skill whenever launching, re-launching, or planning ANY long-context eval
  in this repo — "run evals on <run>", "eval this checkpoint", "score the ladder",
  "launch the xlong rungs", "kick off the eval sweep", or when a
  training run finishes and needs to be scored. It enforces the standing rules
  (OOD always, xlong by bundle, match the checkpoint, ledger every launch) and records
  the results-hub column values the result file cannot know by itself.
---

# Running evals

`records/running-ctc-evals.md` is the authoritative prose on this harness — bundles, tokenizers,
the fast bundle, the traps. **Read it for anything not covered here.** This skill is only the
standing rules and the launch ledger.

## The launcher

```bash
python src/scripts/ctc/eval_beaker.py <RUN_NAME> --tasks main
```

One Beaker job, one checkpoint, every requested task-rung. `<RUN_NAME>` is a directory under
`/weka/oe-training-default/ai2-llm/checkpoints/prasanns/`; the latest complete `stepN` is resolved
**on the node** (`--step stepN` pins one, `--ckpt /abs/path` points anywhere). Results land in
`<run>/ctc_eval/` as one JSON per rung, named `<task>_<rung>_<attn>[_<tag>].json`.

`--priority` defaults to `urgent` and stays there. Cluster defaults to `ai2/jupiter-cirrascale-2`.

**gantry clones the pushed commit, not your working tree.** The launcher hard-errors if HEAD is not
on `origin/<branch>`, so push first. `--dry-run` prints the node command without submitting.

Same command without Beaker (local GPU, staged bundle on node-local `/data`):

```bash
PYTHONPATH=src:ctc/src python -m ctc.eval.cli --ckpt ... --tasks main --bundle /data/.../bundle --out results/<run>
```

(`ctc-eval` is the same entry point once `ctc` is pip-installed.)

## Standing rules

**1. Run the OOD ladders too.** `--tasks main` is the five in-distribution ladders; `--tasks ood` is
fiqa / scifact / outlier_review / contra_fever; `--tasks all` is both. Never report `main` alone as
a complete eval of a checkpoint. `scifact` is eval_size=299 — flag it inline wherever it is quoted.

**2. `--attn`, `--tokenizer` and `--query-position` must match the checkpoint.** Each mismatch
produces a plausible number rather than an error:

- `--attn full|chunked|landmark` — `full` forces plain causal even on a checkpoint carrying the
  mask. The format fingerprint check warns on a mismatch; a checkpoint predating fingerprinting
  needs `--ignore-format-fingerprint`, and the result then records compatibility as UNVERIFIED.
- `--tokenizer` defaults to `Qwen/Qwen3-4B`. A Qwen3.5 checkpoint needs
  `--tokenizer Qwen/Qwen3.5-0.8B-Base` or every token id, EOS included, is wrong.
- `--query-position` defaults to `both`, which is how these checkpoints trained. `after` collapses
  them (nq 0.860 → 0.074). Only change it for a checkpoint trained that way.

**3. xlong is opt-in and bundle-dependent.** `--rungs xlong` is 64k–2M; `--rungs all` is the 2k–32k
ladder and deliberately excludes them. Coverage in the default `v2_clean` bundle is contradiction
and rerank only — **64k+ on nq or outlier requires `--bundle v2`** — and no OOD ladder has xlong
rungs at all. There is no RoPE-extension helper in this repo, so an xlong rung runs at the model's
native positions; say so on any row past the model's ceiling.

**4. Bundles are not interchangeable.** `v2_clean` (default), `v2`, `fast` — `ctc-eval
--list-bundles`. The same rung label maps to different files with different corpus sizes. `fast` is
the shared-corpus construction and is a **different measurement**, not a cheaper route to the same
one (outlier +0.215/+0.261, contradiction −0.102…−0.175); use it to compare arms against each other
at a length you could not otherwise afford, never to fill a cell in a `v2_clean` table.
`--share-prefix` is what turns the KV reuse on and only does anything on `--bundle fast`.

**5. Pass `--tag` whenever two runs could collide.** Result filenames carry task, rung and attn —
**not** the run name or the bundle. A second pass over the same checkpoint (different bundle,
different query-position) lands on the first pass's filenames. Tag by what distinguishes the pass,
e.g. `--tag fast`, `--tag qafter`.

Forgetting is no longer silent: the eval reads the identity of each result file it would write
(bundle, query-position, checkpoint) and **refuses at startup**, naming both values, before the
model loads. A rerun of the *same* pass still overwrites, which is what you want; `--overwrite`
replaces a different pass on purpose, and the usual answer is `--tag`, not that. The results dir
now defaults to the run's own `<ckpt>/../ctc_eval` in every case including `--ckpt`, so two
checkpoints no longer share one directory by default.

**6. Write a ledger entry at submission time** (below).

Worth one minute before a long sweep: `python src/scripts/ctc/eval_beaker.py --check-bundle`
submits a GPU-less job that resolves every rung to a file and reports whether it exists.

## `eval_version` is NOT something you type

Every run writes a `_meta` block into its result JSON — `{ladder_version, eval_bundle,
query_position}` — and results-hub's `ingest_eval_json.py` reads `eval_version` and
`eval_set_weka_pointer` straight out of it. The valid values are `v1`, `v2`, `fast`.

- **Do not record `eval_version` in the ledger and do not pass `--eval-version` on ingest.** A typed
  value that contradicts the run's own `_meta` is a hard error by design.
- The one exception is a legacy result file that predates `_meta` and therefore says nothing about
  which ladder it ran; only then does someone supply the value.
- "v2 is the only supported ladder" was the old instruction and is now wrong: `fast` numbers are a
  separate series in results-hub and are compared to reliable ones deliberately, never merged.

Everything else the run also records for itself — `bundle`, `bundle_root`, `bundle_kind`,
`share_prefix`, `backend`, `eval_size`, `parse_rate`, `standard_errors`, `prompt_tokens`,
`warnings`, and `provenance` (`ckpt`, `git_commit`, `query_position`, `max_length`,
`max_new_tokens`). None of it belongs in the ledger.

## The launch ledger

**Every launch writes one before you report the launch as done.**

```
records/eval_launches/<YYYY-MM-DD>_<run-name>[_<tag>].yaml
```

plus a row in `records/eval_launches/README.md` (date, run, checkpoint, what was launched, link).

**The ledger is local bookkeeping — never commit it.** `records/` is tracked in this repo, so keep
`records/eval_launches/` out of `git add` (and add it to `.gitignore` if it starts showing up in
`git status`). The shared artifact is the row in `../results-hub`; the ledger exists to make those
rows correct and to track which ones still need writing.

It holds exactly two things the result JSON cannot know:

1. **The model-side results-hub columns** — nothing about the model is recorded in the eval output
   beyond the checkpoint path.
2. **The Beaker experiment ID per job**, so `pull-evals` fetches by ID instead of guessing which job
   wrote which file. The launcher prints it:

   ```
   --- ctceval-main-full-<run> ---
   submitted: 01KZQ58X21CM4GPW9R70YK0W5D
   ```

See `ledger_template.yaml` in this skill directory for the shape.

| results-hub column | where it comes from |
|---|---|
| `eval_name` | the ladder — `ladder` in the result JSON |
| `eval_version`, `eval_set_weka_pointer` | the run's `_meta` block. Never typed |
| `eval_context_length`, `eval_data_quantity` | `rung` and `eval_size` in the result JSON |
| `metric_name` / `metric_value` / `extra_metrics` | `primary_metric`, `primary_value`, `metrics` |
| `git_commit` | `provenance.git_commit` — the commit the node ran |
| `decoding_hparams_other` | `backend`, `provenance.max_length` / `max_new_tokens` / `query_position`, `share_prefix` |
| **`model_type`** | dense/normal, landmark, compressive, doc-chunked-landmark — **ledger** |
| **`model_subtype`** | architecture detail, e.g. `block64`, `4lm_mean` — **ledger** |
| **`attention_type`** | `--attn full` → `full`, `--attn chunked` → `doc_chunked` — **ledger** |
| **`pipeline_stage`** | `SFT` / `CPT` / `none` — **ledger** |
| **`chat_template`** | `chat` for SFT checkpoints, `raw` for BASE/CPT — **ledger** |
| **`weka_model_location`** | the resolved `stepN` (the job log prints `CKPT=`) — **ledger** |
| **`model_train_data_hparams`** | a short model slug, e.g. `q4b-compressive-block64-32K` — **ledger** |
| **`other_notes`** | Beaker experiment ID, the full training description, anything making the row non-comparable — **ledger** |

`who_ran` and `date_eval_ran` auto-fill; don't hardcode them. The landmark top-k columns have no
corresponding flag in this harness (`--mem-freq` is the only landmark knob) — leave them blank.

Every job goes in with `status: submitted`. Only `pull-evals` flips it to `done`.

## Scope

The CTC suite via `ctc-eval` only. RULER and HELMET run through a different harness and are out of
scope for this skill.

## Checklist before reporting a launch complete

- [ ] Branch committed **and pushed** (gantry runs the pushed commit)
- [ ] OOD ladders launched, not just `main`
- [ ] `--attn`, `--tokenizer`, `--query-position` match the checkpoint
- [ ] xlong, if wanted, launched with a bundle that actually has those rungs
- [ ] `--tag` distinct for every pass that shares a results directory
- [ ] `--priority urgent`
- [ ] Ledger YAML + README row written, every job carrying its `beaker_experiment_id` and
      `status: submitted`, and **no `eval_version` typed anywhere**
