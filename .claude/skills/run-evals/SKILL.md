---
name: run-evals
description: >
  Use this skill whenever launching, re-launching, or planning ANY long-context eval
  in this repo — "run evals on <run>", "eval this checkpoint", "score the ladder",
  "launch the xlong rungs", "kick off the eval sweep", or when a
  training run finishes and needs to be scored. It enforces the four standing rules
  (xlong always, YaRN by rung, OOD always, ledger every launch) and produces the
  results-hub column values for every job it submits.
---

# Running evals

Every eval launch in this repo is subject to four standing rules. They are not
suggestions and they are not per-request options — apply them unless the user
explicitly overrides one for that launch.

1. **Always run the xlong variants**, not just the base 2k–32k ladder.
2. **YaRN by rung**: `factor=2` for 256k and 512k, `factor=4` for 1M, `factor=8` for 2M.
3. **Always run the OOD ladders** (`fiqa`, `scifact`, `outlier_review`, `contra_fever`).
4. **Always write a launch ledger entry** recording every results-hub column value
   except the metric itself, before/at submission time — not after results land.

## The launcher

Everything below goes through the Beaker launcher:

```
src/scripts/train/memexpress/singletask_ladder/run_q4b_beaker_multirung_eval.py
```

which submits one job per `(run, task)` and drives the on-node runner
`run_beaker_multirung_eval.sh` (8-way DP `torchrun` over
`src/scripts/ctc_eval/eval/eval_lc_native.py`). Launch from the `olmo-core` conda
env, with the branch pushed (gantry checks out a remote ref).

`--priority urgent` is the default and must stay urgent — except on `ai2/holmes`,
which rejects urgent (use `high` there).

**The eval set is standardized.** Every launch reads the clean v2 bundle:

```
/weka/oe-training-default/ai2-llm/checkpoints/prasanns/_eval_bundle_eval500_v2_clean
```

That is the runner's default for `EVAL500` — leave it alone. Do not set `EVAL500` to
the older `_eval_bundle_eval500_v2` unless you are deliberately reproducing a
pre-2026-07-29 number, and say so in the ledger when you do. The two bundles are
byte-identical at ≤32k, but contra at 64k and above is a different, harder task in the
clean bundle (the old one has 28–31% FEVER/wiki distractors, a domain shortcut, and
eval_size=300 at 64k/128k/256k for the doc-pool tasks). Contra numbers must not be
compared across that switch.

**`--query-position` must match the SFT shards.** The v2 ladder rungs and the OOD
probes are raw unified JSONL rendered at eval time, so the prompt layout is a launch
flag, not a property of the eval set:

| training data | flag |
|---|---|
| `xlong5_2k256k_qwen35` (and everything before 2026-08-11) | `both` — the default |
| `xlong5_2k256k_qwen35_qafter` | `after` |

Evaluating a query-after model with `both` hands it a second copy of the task ask it
never saw in training; that reads as a capability gap, not a prompt mismatch. The flag
lands in the inner `eval_command`, so the ledger records it with no extra step — but
numbers are **not comparable across the two settings**, so say which one you used in
`other_notes`. RULER is exempt (never from the 5-task mix; always rendered query-after).

`--ladder-version v2` likewise: v2 is the only supported ladder. v1 resampled questions
per rung, so every rung-to-rung delta carried eval-set noise; the runner rejects it.

## Rule 1 + 3: what gets launched

A complete eval of one checkpoint is **two passes**, because the four OOD ladders have
no xlong rung files (the runner prints `no xlong rungs for TASK=...` and would silently
re-run the base ladder under an xlong tag — a duplicate row, not a new measurement).

**Pass A — base ladder, all 9 tasks (5 in-distribution + 4 OOD):**

```bash
PYTHONPATH=src python src/scripts/train/memexpress/singletask_ladder/run_q4b_beaker_multirung_eval.py \
    <run-name> ai2/jupiter-cirrascale-2 --task all \
    --ckpt /weka/oe-training-default/ai2-llm/checkpoints/<user>/<run>/stepNNNN \
    --eval-tag base
```

**Pass B — xlong rungs, in-distribution tasks only, one submission per YaRN group:**

```bash
# native RoPE (<= 128k): no YaRN copy needed
... --task contra,nq,outlier,rerank,oolong --xlong --xlong-only --xlong-rungs 64k,128k \
    --ckpt <step> --eval-tag xlong-native
```

Never fold the OOD tasks into `--xlong`, and never drop them from Pass A.
`--task all` is the whole 9-task list; if you name tasks explicitly, the OOD four must
still appear in Pass A.

Use `--xlong-only` whenever Pass A has already been submitted. Without it, every xlong
job re-runs 2k–32k first at the bs=1 the xlong path forces — wasted GPU time and a
duplicate of numbers already on weka.

## Rule 2: YaRN factor by rung

Qwen3.5's native ceiling is **262,144 positions**. `eval_lc_native.py` raises the cap
~10% above the rung label, so the realized cap crosses the ceiling starting at the
**256k rung** (the 256k label sits exactly *on* the ceiling; the 0.4–3.3% over-label
overage is what crosses it). A native-RoPE run at 256k is not an in-ceiling
measurement, and it produces a fake collapse — that is what the 2026-08-04 256k sweep
was.

| xlong rungs | checkpoint to eval | YaRN factor |
|---|---|---|
| 64k, 128k | the plain step dir | none |
| 256k, 512k | `<step>_yarn2` | **2** |
| 1M | `<step>_yarn4` | **4** |
| 2M | `<step>_yarn8` | **8** |

Build the serving copies first — they are config-only (~15 KB, symlinks the weights):

```bash
python debug/ctx_ceiling_4b/make_yarn_copy.py --src <step> --factor 2   # -> <step>_yarn2
python debug/ctx_ceiling_4b/make_yarn_copy.py --src <step> --factor 4   # -> <step>_yarn4
```

Then split Pass B by group, pointing `--ckpt` at the matching copy and tagging so the
result files never collide:

```bash
... --xlong --xlong-only --xlong-rungs 256k,512k --ckpt <step>_yarn2 --eval-tag xlong-yarn2
... --xlong --xlong-only --xlong-rungs 1M        --ckpt <step>_yarn4 --eval-tag xlong-yarn4
... --xlong --xlong-only --xlong-rungs 2M        --ckpt <step>_yarn8 --eval-tag xlong-yarn8
```

Do **not** over-scale: one YaRN factor per rung group, never one copy for all rungs —
over-scaling degrades the shorter rungs. Do **not** put a native rung (64k/128k) in the
same submission as a yarn2 rung; a submission has exactly one checkpoint.

`--ngpu` 2 is the default and fine to 128k; 256k+ needs an 80GB GPU
(`ai2/jupiter-cirrascale-2`) and the runner sets `PREFILL_CHUNK_SIZE=32768` itself for
256k and above. Do not remove that — one-shot prefill OOMs past ~256k.

## Rule 4: the launch ledger

**Every launch writes a ledger entry before you report the launch as done.**

Location (consistent, one file per checkpoint being evaluated):

```
records/eval_launches/<YYYY-MM-DD>_<run-name>[_<tag>].yaml
```

and one row appended to `records/eval_launches/README.md` (date, run, checkpoint,
passes launched, link to the YAML).

**The ledger stays out of git.** It is local working bookkeeping, not a repo artifact —
`records/eval_launches/` is gitignored, so don't commit it and don't `git add -f` it.
Share it freely by other means if it's useful to someone; it just doesn't belong in the
history.

The **shared view of any eval is its row in `../results-hub`** — that is the artifact
other people read. The ledger's whole job is to make those rows correct and to track
which ones still need writing.

The YAML records **every results-hub column except `metric_value` / `extra_metrics`**,
so that when the jobs land, ingestion is a mechanical join instead of an archaeology
exercise. See `ledger_template.yaml` in this skill directory for the exact shape.

**Record the Beaker experiment ID for every job.** The launcher prints one line per
task:

```
--- [contra] ev-contra-<run>-<tag>-<hash> ---
    submitted: 01KZQ58X21CM4GPW9R70YK0W5D
```

Both go in the ledger (`beaker_job_name`, `beaker_experiment_id`). The experiment ID is
the load-bearing one: it is how `pull-evals` fetches results and job state directly,
instead of globbing weka and inferring which job wrote which file. A ledger job without
one is barely better than no entry at all.

Every job goes in with `status: submitted`. The `pull-evals` skill is what flips it to
`done` — this skill never marks a job complete.

Column values, and where each one comes from:

| results-hub column | value for this launch |
|---|---|
| `date_eval_ran` | submission date (YYYY-MM-DD) |
| `who_ran` | whoever launched it — results-hub auto-fills from the git user / `$USER`; don't hardcode a name |
| `eval_name` | the task: `contra`/`nq`/`rerank`/`outlier`/`oolong`/`fiqa`/`scifact`/`outlier_review`/`contra_fever` |
| `eval_version` | `v2` (the only supported ladder; v1 is rejected by the runner) |
| `eval_context_length` | the rung — one row per rung, not one per job |
| `eval_data_quantity` | realized `eval_size` per rung (from the result JSON; **not** `--max-test`) |
| `eval_set_weka_pointer` | always the standard bundle (below) unless you deliberately overrode `EVAL500` |
| `metric_name` | `f1` for contra/nq/fiqa/scifact/contra_fever, `ndcg@10` for rerank, task-native for oolong/outlier |
| `model_type` | dense / landmark / compressive / docchunk (matches the inferred `VARIANT`) |
| `model_subtype` | architecture detail, e.g. `block64`, `4lm_mean` |
| `attention_type` | e.g. `full`, `compressive_landmark`, `doc_chunked` |
| `pipeline_stage` | `SFT` / `CPT` / `base` |
| `weka_model_location` | the `--ckpt` **as passed** — i.e. the `_yarn2` / `_yarn4` path when one was used |
| `model_train_data_hparams` | a **short model slug**, e.g. `q35-landmark-256K-partialrope` — one per ledger file, shared by the native *and* YaRN passes (same weights). `pull-evals` derives it and records it as `checkpoint.model_slug`. The full training description (data mix, steps, GBS, lr, parallelism, base init) goes in the ledger's `training_description` and ends up in `other_notes` |
| `landmark_top_k_fixed_val` / `landmark_top_k_percentage` / `landmark_nonselected_percentage` | the decode knobs actually passed; blank = checkpoint default (10%-of-prompt top-k, trained mass) |
| `decoding_hparams_other` | temp/top_p/`max_length` (the **realized** cap, after the 10% raise), `batch_size`, `ngpu`, **and the YaRN factor** |
| `chat_template` | `chat` for SFT checkpoints, `raw` for BASE/CPT |
| `git_commit` | repo HEAD at submission |
| `eval_command` | the full `RUN=... TASK=... bash run_beaker_multirung_eval.sh` inner command the launcher prints |
| `other_notes` | Beaker job id + job name, the full training description, YaRN factor and why, and anything that makes the row non-comparable |

Two traps to honor when filling this in:

- **`batch_size` in `eval_command` is not what ran.** The runner forces `BATCH_SIZE=1`
  for `landmark`/`compressive`, and the xlong path forces bs=1 for everyone. Record what
  actually ran in `decoding_hparams_other`.
- **Any run at ≥256k must be labeled RoPE-extended** in `other_notes`
  (`yarn_factor=2`, etc.). A ≥256k number without that label is unusable — nobody can
  tell later whether it was an in-ceiling measurement.

## Scope

This skill covers the multi-rung native long-context ladder only. RULER and HELMET run
through a different harness with its own conventions and are **out of scope** — do not
launch them from this skill, and do not treat the rules above as applying to them.

## After the jobs land

Not this skill's job. Use **`pull-evals`**, which cross-references the finished jobs
against the ledger, checks the generations before anything is believed, ingests into
`../results-hub`, and marks each job `done` in the ledger.

## Checklist before reporting a launch complete

- [ ] Pass A submitted with all 9 tasks (OOD included)
- [ ] Pass B submitted with `--xlong --xlong-only`, split by YaRN group
- [ ] YaRN copies built for every rung ≥256k: factor 2 (256k/512k) / 4 (1M) / 8 (2M)
- [ ] Distinct `--eval-tag` per pass and per YaRN group (no result-file collisions)
- [ ] `--priority urgent` (or `high` on holmes)
- [ ] Tokenizer matches the model family (Qwen3.5 → `Qwen/Qwen3.5-0.8B`)
- [ ] Eval set is the standard `_eval_bundle_eval500_v2_clean` bundle at ladder `v2`
- [ ] Ledger YAML written under `records/eval_launches/` + README row appended, every
      job carrying its `beaker_experiment_id`, job name and `status: submitted`
