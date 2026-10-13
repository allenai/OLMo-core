---
name: run-evals
description: >
  Use this skill whenever launching, re-launching, or planning ANY long-context eval
  in this repo — "run evals on <run>", "eval this checkpoint", "score the ladder",
  "launch the xlong rungs", "kick off the eval sweep", or when a
  training run finishes and needs to be scored. It enforces the four standing rules
  (xlong always, YaRN by rung, OOD always, register every launch) and registers every
  eval job with `ops register --kind eval --parent-run <train run>`.
---

# Running evals

Every eval launch in this repo is subject to four standing rules. They are not
suggestions and they are not per-request options — apply them unless the user
explicitly overrides one for that launch.

1. **Always run the xlong variants**, not just the base 2k–32k ladder.
2. **YaRN by rung**: `factor=2` for 256k and 512k, `factor=4` for 1M, `factor=8` for 2M.
3. **Always run the OOD ladders** (`fiqa`, `scifact`, `outlier_review`, `contra_fever`).
4. **Always register the launch** at submission time, not after results land:
   `ops register --kind eval --parent-run <train run> --job-id <beaker id> --expected-minutes N --log-path ...`
   (see Rule 4 below). The YAML ledger in `records/eval_launches/` is optional history.

## CTC-suite evals: use the olmo-eval launcher (current, 2026-10)

For the CTC suite (the 22-row roster + OOD rows, 2k-256k) the current path is **olmo-eval's
`scripts/ctc_suite/launch_ctc_suite.py`** (olmo-eval `main` once PR #467 merges; until then branch
`prasann/ctc-suite-launcher-on-main`; README in that folder; default `--rows roster`). It packs single-GPU Beaker jobs on jupiter and runs a per-job `preflight.py`. Its defaults
already apply rule 2 at the r256k rung: **YaRN factor 2 + 131,072-token chunked prefill**, in jobs of
their own, with every other rung untouched. You need no `_yarn2` copy; the model runs from the step
dir. That rung's prompts run to ~272k tokens, and without YaRN 43-100% of rows are left-truncated
(chat header + instruction cut).

```bash
uv run python scripts/ctc_suite/launch_ctc_suite.py --ckpt <weka step dir> --arm full \
    --doc-markers --rows <list> --run-name <name>        # --dry-run first
```

- **Tokenizer:** keep the default `Qwen/Qwen3.5-0.8B`. The 4B tokenizer's chat template opens a
  `<think>` block for the generation prompt, which is out of format for every CTC SFT checkpoint.
- **`--doc-markers`:** pass it for checkpoints trained on marker-wrapped shards (e.g. setA).
- **`--fit-window`:** the r256k alternative to YaRN. It shortens distractor documents and agreed
  with YaRN within noise on the setA pair.

The multirung launcher below remains the path for the singletask/xlong ladders it was written for.

## The launcher (singletask / xlong ladders)

Everything below goes through the Beaker launcher:

```
recipes/eval/legacy_native/run_q4b_beaker_multirung_eval.py
```

which submits one job per `(run, task)` and drives the on-node runner
`run_beaker_multirung_eval.sh` (8-way DP `torchrun` over
`recipes/eval/legacy_native/ctc_eval/eval/eval_lc_native.py`). Launch from the `olmo-core` conda
env, with the branch pushed (gantry checks out a remote ref).

`--priority urgent` is the default and must stay exactly `urgent` (never lower, never
`immediate`). `ai2/holmes` rejects urgent, so don't launch there — use jupiter.

**The eval set is standardized.** Every launch reads the clean v2 bundle:

```
/weka/oe-training-default/ai2-llm/checkpoints/prasanns/_eval_bundle_eval500_v2_clean
```

That is the runner's default for `EVAL500` — leave it alone. Do not set `EVAL500` to
the older `_eval_bundle_eval500_v2` unless you are deliberately reproducing a
pre-2026-07-29 number, and say so in the `--notes` of `ops register` when you do. The two bundles are
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
lands in the inner `eval_command`, so put it in the `ops register --notes` too — but
numbers are **not comparable across the two settings**, so say which one you used in
the `--notes`. RULER is exempt (never from the 5-task mix; always rendered query-after).

`--ladder-version v2` likewise: v2 is the only supported ladder. v1 resampled questions
per rung, so every rung-to-rung delta carried eval-set noise; the runner rejects it.

## Rule 1 + 3: what gets launched

A complete eval of one checkpoint is **two passes**, because the four OOD ladders have
no xlong rung files (the runner prints `no xlong rungs for TASK=...` and would silently
re-run the base ladder under an xlong tag — a duplicate row, not a new measurement).

**Pass A — base ladder, all 9 tasks (5 in-distribution + 4 OOD):**

```bash
PYTHONPATH=src python recipes/eval/legacy_native/run_q4b_beaker_multirung_eval.py \
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
python recipes/eval/make_yarn_copy.py --src <step> --factor 2   # -> <step>_yarn2
python recipes/eval/make_yarn_copy.py --src <step> --factor 4   # -> <step>_yarn4
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

## Rule 4: register every launch

**Every eval job is registered with `ops` before you report the launch as done**
(`OPS=/accounts/projects/berkeleynlp/prasann/bin/ops`; details in the `launch-run` skill). Each eval
pass is its own run: `run_id` = the launcher's natural eval name (`run_q4b_beaker_multirung_eval.py`
registers its jobs itself as `ev-<task>-<run>`, timestamp-suffixed when that name is taken, and
exports it to the job as `OPS_EVAL_RUN_ID`), else `<train_run>__ev-<suite_or_task>-<variant>-<YYYYmmddHHMM>`
(variant = e.g. `yarn2`, `chunked`, `doc-markers`); `--parent-run` = the training run scored.
Result rows then record under that eval run_id with `train_run` = the parent. By hand:

```bash
$OPS register --run-id <train run>__ev-<task>-<variant>-<YYYYmmddHHMM> --experiment <E> --cluster beaker --kind eval \
  --parent-run <train run_id> --task <task> --job-id <beaker experiment id> \
  --ckpt-path <--ckpt as passed, incl. _yarn2> --expected-minutes N --git <repo> \
  --notes "eval-tag=<tag> yarn_factor=<f|none> query_position=<both|after> bs=<what ran>"
```

**Record the Beaker experiment ID for every job.** The launcher prints one line per task:

```
--- [contra] ev-contra-<run>-<tag>-<hash> ---
    submitted: 01KZQ58X21CM4GPW9R70YK0W5D
```

That ID is `--job-id`: it is how `ops status` / `ops stall` find the job and how the
`pull-evals` / `record-results` skills match results to it, instead of globbing weka and
inferring which job wrote which file. Check on the jobs with the `job-status` skill
(`ops status --experiment E`), never a sleep loop.

The YAML ledger (`records/eval_launches/<YYYY-MM-DD>_<run-name>[_<tag>].yaml`, shape in
`ledger_template.yaml`, gitignored — never `git add -f` it) is optional history now; write
it only if you want the extra launch detail. results-hub is retired (2026-10-12): numbers
are recorded with `ops result add --check` (the `record-results` skill), not ingested there.

Two traps that belong in `--notes` (and in the result row later):

- **`batch_size` in `eval_command` is not what ran.** The runner forces `BATCH_SIZE=1`
  for `landmark`/`compressive`, and the xlong path forces bs=1 for everyone. Record what
  actually ran.
- **Any run at ≥256k must be labeled RoPE-extended** (`yarn_factor=2`, etc.). A ≥256k
  number without that label is unusable — nobody can tell later whether it was an
  in-ceiling measurement.

## Scope

This skill covers the multi-rung native long-context ladder only. RULER and HELMET run
through a different harness with its own conventions and are **out of scope** — do not
launch them from this skill, and do not treat the rules above as applying to them.

## After the jobs land

Not this skill's job. Use **`pull-evals`**, which cross-references the finished jobs
against the registry (`ops runs`), checks the generations before anything is believed,
and records the numbers via the **`record-results`** skill (`ops result add --check`).

## Checklist before reporting a launch complete

- [ ] Pass A submitted with all 9 tasks (OOD included)
- [ ] Pass B submitted with `--xlong --xlong-only`, split by YaRN group
- [ ] YaRN copies built for every rung ≥256k: factor 2 (256k/512k) / 4 (1M) / 8 (2M)
- [ ] Distinct `--eval-tag` per pass and per YaRN group (no result-file collisions)
- [ ] `--priority urgent` (exactly urgent; never holmes)
- [ ] Tokenizer matches the model family (Qwen3.5 → `Qwen/Qwen3.5-0.8B`)
- [ ] Eval set is the standard `_eval_bundle_eval500_v2_clean` bundle at ladder `v2`
- [ ] Every job registered (`ops register --kind eval --parent-run ... --job-id <beaker id>`);
      `ops status --experiment E` shows them
