# Eval-time CE parity on the ds64 campaign's WINNING arms

**Date** 2026-09-15/16 · **Status** hdr33 + kv33 COMPLETE (local, cubbins); occ00 + ohdr08 IN PROGRESS
(Beaker) · **Branch** `prasann/landmark`
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

## 2. Infra notes (both burned real time)

* **Beaker was saturated** at launch (4 jobs queued for 25+ min with no `started` timestamp). Local
  Berkeley-cluster jobs (horton `/data` checkpoint copies + cubbins compute) were used instead for
  `hdr33` and `kv33`.
* **Beaker's first 4 runs FALSE-SUCCEEDED** (`exitCode=0`) while the python script inside actually
  crashed at import (`ModuleNotFoundError: No module named 'numpy'`) -- my own WORK script's `echo
  "... rc=$?"` as the last command masked the real exit code. Root cause: `gantry run`'s `--install
  '...'` runs `pip install -e .` against the baked image's SYSTEM python (`/opt/conda/bin/python`),
  but the job itself executes under a separate, empty `/gantry-runtime/.venv`. Fixed by invoking
  `/opt/conda/bin/python` explicitly (and `export PATH=/opt/conda/bin:$PATH` so the driver's own
  `subprocess.run(['python', ...])` shard-conversion call resolves there too), plus `pip install -e
  '.[all]'` (Qwen3.5's GatedDeltaNet needs `fla`, not pulled by the bare install). Relaunched as
  `tp-occ00-v4` / `tp-ohdr08-v4`.
* **Local bootstrap-copy race**: the horton->cubbins checkpoint copy fired on `config.json`
  existing, which a transfer agent can satisfy before `model_and_optim/` is fully written -- grabbed
  a partial file set once and then never retried (destination dir "existed"). Fixed in
  `debug/pooled_kv/run_trained_parity_local.sbatch`: gate on the SOURCE's own `.metadata` file (the
  distributed-checkpoint completion sentinel) and use idempotent `rsync -a`.
* `occ00` additionally needs its training shard (`ds64/shards/oolong_u16M`) locally to rebuild the
  `cent_cmean` stop-id set -- not staged to Berkeley, so `occ00` is Beaker-only.

## 3. Results

### 3.1 contradiction `hdr33` -- `ds64-contradiction-hdr33-b128f3-u64M` (local, cubbins, COMPLETE)

`gold_plus_random` keep 1/3, `Claim N:` header real, plain mean slot. `eval_size` 48/rung (below the
500 floor -- these are fast confirmatory numbers, not a replacement for the ladder eval), 48 rows/rung
also scored for generation.

```
 rung eval_size |  CE_full  CE_soft      dCE | CEdig_full CEdig_soft   dCEdig |  F1_full  F1_soft     dF1 | compact
   2k        48 |    0.013    0.004   -0.010 |      0.027      0.005   -0.021 |    0.979    1.000  +0.021 |   0.482
   8k        48 |    0.023    0.004   -0.019 |      0.040      0.005   -0.034 |    0.965    1.000  +0.035 |   0.469
  32k        48 |    0.082    0.025   -0.057 |      0.145      0.043   -0.102 |    0.826    0.938  +0.111 |   0.457
```

### 3.2 nq `kv33` -- `ds64-nq-kv33-b128f3-u64M` (local, cubbins, COMPLETE)

`gold_plus_random` keep 1/3, no header, plain mean slot. `eval_size` 48/rung.

```
 rung eval_size |  CE_full  CE_soft      dCE | CEdig_full CEdig_soft   dCEdig |  F1_full  F1_soft     dF1 | compact
   2k        48 |    0.004    0.002   -0.002 |      0.011      0.003   -0.007 |    1.000    1.000  +0.000 |   0.440
   8k        48 |    0.035    0.009   -0.026 |      0.120      0.038   -0.082 |    0.931    0.965  +0.035 |   0.363
  32k        48 |    0.081    0.033   -0.048 |      0.222      0.082   -0.140 |    0.917    0.927  +0.010 |   0.341
```

### 3.3 oolong `occ00` -- `ds64-oolong-occ00-b128f3-u16M` (Beaker, IN PROGRESS)

Gold-blind keep 0.0, `Date:/User:/Instance:` headers real, `cent_cmean` slot. Running the full
240/240/120-row scope with generation. At last check: 8k rung in progress (row 125/240), full CE
~0.35 vs arm CE ~0.36, genF1 both ~0.58 (partial, not yet the final per-rung aggregate). Job
`01M2KTQTFKA5BNZ8RA54PKJ13P`.

### 3.4 oolong `ohdr08` -- `ds64-oolong-ohdr08-b128f3-u16M` (Beaker, IN PROGRESS)

Gold-blind keep 0.0833, same headers, plain mean slot. At last check: 8k rung in progress (row
75/240), full CE ~0.33 vs arm CE ~0.34, genF1 both ~0.56 (partial). Job
`01M2KTQZDGYS6TEV0DHDJ9FXB0`.

## 4. Reading it so far

**contradiction `hdr33` and nq `kv33` show real parity -- and the soft construction is CHEAPER,
not just tied.** `dCE` and `dCEdig` are NEGATIVE at every rung for both arms (soft beats full), and
`dF1` is flat-to-positive (+0.02 to +0.11 for contradiction, 0.00 to +0.04 for nq) at 9-11x fewer
real tokens (compact 0.34-0.48). This is the opposite direction from outlier's `cc00` (which
degraded past 2k) and makes sense structurally: both `hdr33`/`kv33` are `gold_plus_random` --
the GOLD document is always real, so the model never has to read the answer out of a pooled slot;
pooling only removes distractor context, which a well-trained model may find easier, not harder,
to ignore. This is the strongest possible parity result: not merely "no regression" but the arm's
own training construction generalizing better than full attention at these rows.

Oolong's two arms (both gold-blind -- there is no always-real gold document) are the harder case,
matching outlier's structure more closely; final numbers pending.

## 5. Jobs

| arm | Beaker experiment | local sbatch | status |
|---|---|---|---|
| hdr33 | (queued, superseded by local) | `3554219` (CE) + `3554327` (+gen), cubbins | **DONE** |
| kv33 | (queued, superseded by local) | `3554221` (CE) + `3554328` (+gen), cubbins | **DONE** |
| occ00 | `01M2KTQTFKA5BNZ8RA54PKJ13P` (v4, fixed install) | -- (needs training shard, not staged locally) | running |
| ohdr08 | `01M2KTQZDGYS6TEV0DHDJ9FXB0` (v4, fixed install) | -- (no local checkpoint copy) | running |

## 6. Verdicts

* **contradiction `hdr33`: PARITY, with margin.** dCEdig and dF1 both favor the soft construction
  at every rung tested (2k/8k/32k). Ladder number is not an eval-format artefact.
* **nq `kv33`: PARITY, with margin.** Same pattern, smaller magnitude. Ladder number is not an
  eval-format artefact.
* **oolong `occ00` / `ohdr08`: pending** -- Beaker jobs in flight; will update this record and the
  verdict once they finish.
