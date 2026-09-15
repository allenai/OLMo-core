# Eval-time CE parity on the ds64 campaign's WINNING arms

**Date** 2026-09-15 · **Status** IN PROGRESS · **Branch** `prasann/landmark`
**Driver** `debug/pooled_kv/trained_parity_check.py` (new; generalizes
`debug/pooled_kv/outlier_probe/outlier_slot_probe.py --trained-parity` beyond outlier -- see its
module docstring for why: that probe refuses `--st-keep-frac`/`gold_plus_random`, which is exactly
what contradiction's `hdr33` and nq's `kv33` train with, and grades free generation with outlier's
own bracket-id set-F1 instead of each task's own grader)
**Context** `records/outlier-cc00-evaltime-probe.md` (the check this extends to the campaign's other
winning arms), `records/ds64-handoff.md` (the campaign's standing frontier per task).

## 1. Why

Every number in the ds64 campaign is a ladder eval: the checkpoint is fed FULL real text at every
rung. A checkpoint trained on a soft-token construction (pooled documents, a header kept real, a
gold-blind or gold-plus-random keep policy) has never seen that full-real-text input during
training, so the ladder number silently conflates two different things: "the compression transfers
to eval" and "the compression method itself works." `outlier-cc00-evaltime-probe.md` answered this
for outlier's `cc00` (verdict: does not transfer past 2k). This record runs the same check -- same
checkpoint, twice per rung, once on FULL real text and once on its own TRAINING construction -- on
the campaign's actual Pareto-frontier arms, so those wins are confirmed as a genuine compression
result rather than an eval-format artefact.

## 2. Arms checked

| task | arm | checkpoint | construction |
|---|---|---|---|
| oolong | `occ00` | `ds64-oolong-occ00-b128f3-u16M` | gold-blind keep 0.0, `Date:/User:/Instance:` headers real (`--st-header-stop-id 25 --st-header-stop-count 3`), **`cent_cmean`** slot |
| oolong | `ohdr08` | `ds64-oolong-ohdr08-b128f3-u16M` | same, but keep 0.0833 and the **plain mean** slot |
| contradiction | `hdr33` | `ds64-contradiction-hdr33-b128f3-u64M` | `gold_plus_random` keep 1/3, `Claim N:` header real (`--st-header-stop-id 25 --st-header-stop-count 1`), plain mean slot |
| nq | `kv33` | `ds64-nq-kv33-b128f3-u64M` | `gold_plus_random` keep 1/3, **no header**, plain mean slot |

Rungs 2k / 8k / 32k, `eval_size` 240 / 240 / 120 (>=200 at 2k/8k, >=100 at 32k per the standing
size floor), free-generation on 48 / 48 / 24 rows/rung, each task graded with its own metric
(contradiction: pair-F1 via `ctc_eval`'s own `_eval_contradiction`; nq: retrieval-F1; oolong: its
own EM / numeric-partial-credit / set-F1, driven off the row's own decoded answer span -- see the
driver's module docstring for why grading never touches the gold sidecar's per-task index-base
convention).

## 3. Infra note

Beaker (`ai2/ceres-cirrascale`, `ai2/saturn-cirrascale`, `ai2/jupiter-cirrascale-2`) was saturated
at launch time (all four jobs sat `queued`, no `started` timestamp, for the first several minutes).
In parallel, model-only checkpoint copies of `occ00`/`hdr33`/`kv33` (not `ohdr08`) were staged to
horton `/data/prasann/ckpts/<run>/` with eval-rung mirrors under `/data/prasann/ds64_eval/<task>/`,
and `debug/pooled_kv/run_trained_parity_local.sbatch` (waits for its own checkpoint to land, then
runs locally, `--qos=preemptive_high_sewonm` on horton) was submitted for `hdr33` and `kv33`. `occ00`
needs its training shard (`ds64/shards/oolong_u16M`) for the `cent_cmean` stop-id set, which was NOT
staged locally, so it stays Beaker-only; `ohdr08` (no local checkpoint copy) also stays Beaker-only.
Whichever path lands first for a given arm is what is reported below; the other is cancelled.

## 4. Jobs

| arm | Beaker experiment | local sbatch | status |
|---|---|---|---|
| occ00 | `01M2K959SQ4SA1ZVQ8GPQZ74KA` | -- (needs training shard, Beaker only) | pending |
| ohdr08 | `01M2K961NZ9N76B6D69887EC2D` | -- (no local checkpoint copy) | pending |
| hdr33 | `01M2K95HMFSDXMKN38YY4AD6A1` | `3552619` (horton) | pending |
| kv33 | `01M2K95NEFA4KRH48FVRDVVY8B` | `3552620` (horton) | pending |

## 5. Results

_(filled in as each arm's JSON lands)_

## 6. Verdicts

_(filled in once results land: parity = `|dCEdig|` within ~1 SE of 0 AND the generation metric gap
within noise, per the reading rule in `outlier-cc00-evaltime-probe.md` §2)_
