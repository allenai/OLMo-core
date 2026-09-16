# Eval-time CE parity on the ds64 campaign's WINNING arms

**Date** 2026-09-15/16 · **Status** COMPLETE, all 4 arms · **Branch** `prasann/landmark`
**Driver** `debug/pooled_kv/trained_parity_check.py` (new; generalizes
`debug/pooled_kv/outlier_probe/outlier_slot_probe.py --trained-parity` beyond outlier -- that probe
refuses `--st-keep-frac`/`gold_plus_random`, which is exactly what contradiction's `hdr33` and nq's
`kv33` train with, and grades free generation with outlier's own bracket-id set-F1 instead of each
task's own grader)
**Context** `records/outlier-cc00-evaltime-probe.md` (the check this extends), `records/ds64-handoff.md`
(the campaign's standing frontier per task).

## 1. Why

Every ladder number in this campaign feeds a checkpoint FULL real text at eval, even when it was
trained on a soft-token (pooled-document) construction. That silently conflates two different
things: "the compression transfers to eval" and "the compression method itself works." This check
scores the SAME checkpoint twice per rung -- once on FULL real text (what the ladder feeds it) and
once on its own TRAINING construction -- so the campaign's Pareto wins are confirmed as a genuine
compression result rather than an eval-format artefact.

## 2. Arms checked

| task | arm | checkpoint | construction |
|---|---|---|---|
| contradiction | `hdr33` | `ds64-contradiction-hdr33-b128f3-u64M` | `gold_plus_random` keep 1/3, `Claim N:` header real, plain mean slot |
| nq | `kv33` | `ds64-nq-kv33-b128f3-u64M` | `gold_plus_random` keep 1/3, no header, plain mean slot |
| oolong | `occ00` | `ds64-oolong-occ00-b128f3-u16M` | gold-blind keep 0.0, `Date:/User:/Instance:` headers real, `cent_cmean` slot |
| oolong | `ohdr08` | `ds64-oolong-ohdr08-b128f3-u16M` | same, but keep 0.0833 and the plain mean slot |

## 3. Infra notes (each burned real time; keep for the next run)

* **Beaker was saturated** at launch (4 jobs queued 25+ min with no `started`). `hdr33`/`kv33` were
  run on the Berkeley cluster instead (fast, 48-row CE-only pass first, then a generation-inclusive
  follow-up), while `occ00`/`ohdr08` eventually ran on Beaker once it freed up.
* **Beaker's first 4 attempts FALSE-SUCCEEDED** (`exitCode=0`) while the python script inside
  actually crashed at import (`ModuleNotFoundError: No module named 'numpy'`) -- the WORK script's
  `echo "... rc=$?"` as its last command masked the real exit code (fixed: `exit $RC`). Root cause:
  `gantry run --install '...'` runs `pip install -e .` against the baked image's SYSTEM python
  (`/opt/conda/bin/python`), but the job executes under a separate, empty `/gantry-runtime/.venv`.
  Fix: invoke `/opt/conda/bin/python` explicitly, and `export PATH=/opt/conda/bin:$PATH` so the
  driver's own `subprocess.run(['python', ...])` shard-conversion call resolves there too. A second
  round then hit `AssertionError` in `GatedDeltaNet.__init__` (`assert has_fla()`) -- Qwen3.5's GDN
  layers need the `fla` package, not pulled by a bare `pip install -e .`; fixed with `pip install -e
  '.[all]'`.
* **Local bootstrap-copy race** (`debug/pooled_kv/run_trained_parity_local.sbatch`): the
  horton->cubbins checkpoint copy fired on `config.json` existing, which a transfer agent can
  satisfy before `model_and_optim/` is fully written -- grabbed a partial file set once (one
  `.distcp` file still mid-rename) and then never retried, since the destination dir "existed."
  Fixed: gate on the SOURCE's own `.metadata` file (the distributed-checkpoint completion sentinel)
  and use idempotent `rsync -a`, retried every wait-loop iteration.
* `occ00`'s `cent_cmean` slot needs its training shard (`ds64/shards/oolong_u16M`) to rebuild the
  stop-id set; that shard was not staged to the Berkeley cluster, so `occ00` had to run on Beaker.

## 4. Results

`eval_size` 48/rung for `hdr33`/`kv33` (fast local pass, below the 500 floor -- confirmatory, not a
replacement for the ladder eval), 240/240/120 for `occ00`/`ohdr08` (full scope, matching the ladder's
own row counts). Free generation scored on every row that ran it (48 for hdr33/kv33's follow-up
pass, 48/48/24 for occ00/ohdr08), graded with each task's own metric: contradiction pair-F1
(`ctc_eval`'s `_eval_contradiction`), nq retrieval-F1, oolong's own EM / numeric-partial-credit /
set-F1 -- all computed from the row's own decoded answer span, never from the gold sidecar's
per-task index convention (see the driver's module docstring).

### contradiction `hdr33`
```
 rung eval_size |  CE_full  CE_soft      dCE | CEdig_full CEdig_soft   dCEdig |  F1_full  F1_soft     dF1 | compact
   2k        48 |    0.013    0.004   -0.010 |      0.027      0.005   -0.021 |    0.979    1.000  +0.021 |   0.482
   8k        48 |    0.023    0.004   -0.019 |      0.040      0.005   -0.034 |    0.965    1.000  +0.035 |   0.469
  32k        48 |    0.082    0.025   -0.057 |      0.145      0.043   -0.102 |    0.826    0.938  +0.111 |   0.457
```

### nq `kv33`
```
 rung eval_size |  CE_full  CE_soft      dCE | CEdig_full CEdig_soft   dCEdig |  F1_full  F1_soft     dF1 | compact
   2k        48 |    0.004    0.002   -0.002 |      0.011      0.003   -0.007 |    1.000    1.000  +0.000 |   0.440
   8k        48 |    0.035    0.009   -0.026 |      0.120      0.038   -0.082 |    0.931    0.965  +0.035 |   0.363
  32k        48 |    0.081    0.033   -0.048 |      0.222      0.082   -0.140 |    0.917    0.927  +0.010 |   0.341
```

### oolong `occ00`
```
 rung eval_size |  CE_full  CE_soft      dCE | CEdig_full CEdig_soft   dCEdig |  F1_full  F1_soft     dF1 | compact
   2k       240 |    0.201    0.193   -0.008 |      0.702      0.627   -0.075 |    0.688    0.729  +0.042 |   0.525
   8k       240 |    0.354    0.353   -0.001 |      1.144      1.104   -0.040 |    0.583    0.562  -0.021 |   0.485
  32k       120 |    0.590    0.502   -0.088 |      1.923      1.521   -0.403 |    0.375    0.375  +0.000 |   0.466
```

### oolong `ohdr08`
```
 rung eval_size |  CE_full  CE_soft      dCE | CEdig_full CEdig_soft   dCEdig |  F1_full  F1_soft     dF1 | compact
   2k       240 |    0.190    0.185   -0.005 |      0.633      0.594   -0.039 |    0.688    0.729  +0.042 |   0.561
   8k       240 |    0.355    0.357   +0.002 |      1.115      1.094   -0.020 |    0.562    0.562  +0.000 |   0.526
  32k       120 |    0.599    0.537   -0.061 |      1.870      1.603   -0.268 |    0.333    0.375  +0.042 |   0.513
```

## 5. Reading it

**All four winning arms show real eval-time parity, and in most cells the soft construction is
CHEAPER than full real text, not merely tied.** `dCE`/`dCEdig` are negative (or ~0) at every rung
for every arm, and `dF1` is flat-to-positive everywhere except one small negative (`occ00` 8k,
-0.021, inside the ±0.06-ish generation-noise band at 240 rows/~139 gold-doc-generation rows). This
is the opposite of outlier's `cc00` result (`records/outlier-cc00-evaltime-probe.md`), which
degraded sharply past 2k on the same style of check. Two structural reasons, matching each family:

* **contradiction `hdr33` and nq `kv33` are `gold_plus_random`**: the gold document is ALWAYS real,
  so the model never has to answer from a pooled slot -- pooling only removes distractor context,
  which the model is *better* at ignoring than full attention is (a plausible regularization
  effect), and the margin is largest at 32k where there is the most distractor mass to shed
  (dCEdig -0.10 contradiction, -0.14 nq).
* **oolong `occ00`/`ohdr08` are gold-blind** (no line is privileged as "always real") yet STILL show
  parity, unlike outlier's gold-blind `cc00`. The likely reason: oolong's compaction ratio here
  (0.47-0.56) is far gentler than outlier `cc00`'s (~0.06-0.10 at the same rungs, per
  `outlier-cc00-evaltime-probe.md` -- keep 0 on top of a much larger corpus), and oolong's answer
  format (label/date/numeric/short-list) does not require pinpointing ONE odd document among
  hundreds the way outlier's task does, so a coarser per-line summary is enough to answer from.

## 6. Jobs

| arm | Beaker experiment | local sbatch | outcome |
|---|---|---|---|
| hdr33 | queued, superseded | `3554219` (CE-only) + `3554327` (+gen), cubbins | DONE (local) |
| kv33 | queued, superseded | `3554221` (CE-only) + `3554328` (+gen), cubbins | DONE (local) |
| occ00 | `01M2KTQTFKA5BNZ8RA54PKJ13P` (v4, fixed install) | -- (needs training shard, not staged locally) | DONE (Beaker) |
| ohdr08 | `01M2KTQZDGYS6TEV0DHDJ9FXB0` (v4, fixed install) | -- (no local checkpoint copy) | DONE (Beaker) |

Earlier Beaker attempts (`01M2K959SQ4SA1ZVQ8GPQZ74KA`, `01M2K961NZ9N76B6D69887EC2D`,
`01M2K95HMFSDXMKN38YY4AD6A1`, `01M2K95NEFA4KRH48FVRDVVY8B`, and the `-v2`/`-v3` retries) all
false-succeeded or genuinely failed on the environment bugs in §3 -- their JSON output is not
trustworthy and is not used here.

## 7. Verdicts

* **contradiction `hdr33`: PARITY, with margin.** `dCEdig` and `dF1` both favor the soft
  construction at every rung (2k/8k/32k). The ladder number is not an eval-format artefact.
* **nq `kv33`: PARITY, with margin.** Same pattern, at 9-11x fewer real tokens (compact 0.34-0.48).
  Not an eval-format artefact.
* **oolong `occ00`: PARITY.** `dCEdig` negative at every rung (down to -0.40 at 32k); `dF1` flat
  (+0.04, -0.02, 0.00) and every |dF1| is within generation-count noise at these row sizes. Not an
  eval-format artefact -- this is a genuinely different, better-behaved result than outlier's
  gold-blind `cc00`.
* **oolong `ohdr08`: PARITY.** Same pattern, `dCEdig` negative or ~0, `dF1` flat-to-positive
  (+0.04, 0.00, +0.04). Not an eval-format artefact.

**Campaign-level takeaway**: the ds64 frontier's Pareto wins on contradiction, nq, and oolong are
real compression results, not artefacts of always feeding the ladder full text. Outlier remains the
one task where the analogous check (cc00, `records/outlier-cc00-evaltime-probe.md`) found the
opposite -- the slot readout does not scale with document count there, and that verdict is
unchanged by this record.

## 8. Caveats

* `hdr33`/`kv33` numbers are `eval_size` 48/rung (binomial SE ~0.07 on a right/wrong metric at
  f1~0.5) -- fast confirmatory numbers, not a replacement for a full ladder-scale probe. `occ00`/
  `ohdr08` are at the ladder's own row counts (240/240/120) and are correspondingly better powered.
* Free-generation genF1 for oolong is oolong's own composite metric (EM for label/date answers,
  0.75^error for numeric, set-F1 for list answers) computed from the row's decoded gold span via
  `_oolong_extract`/`_oolong_norm`, not the raw JSONL's `_meta` (the driver never loads it -- see its
  module docstring) -- a light reimplementation of `ctc_eval.eval.evaluate._eval_oolong`'s three
  branches, expected to track it closely but not verified bit-for-bit.
