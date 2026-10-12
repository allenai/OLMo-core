# CTC suite → olmo-eval handoff (2026-08-14, refreshed 2026-09-08)

The 22-task suite lives on the `prasann/ctc-suite` branch of allenai/olmo-eval (head `a31c9ce`);
the reviewer-facing doc is `src/olmo_eval/evals/tasks/ctc_suite/README.md` on that branch, the
build provenance is `debug/ctc_1m_ladders/REPORT.md` here, and the data is the public HF dataset
`PrasannSinghal/ctc-suite-eval`. Below is the ready-to-post PR body.

---

# PR draft: `prasann/ctc-suite` → allenai/olmo-eval

**Title:** CTC suite: 22 long-context tasks with context ladders from 2k to 1M tokens

## Summary

Adds the CTC (corpus-tracking-capacity) long-context family: 22 tasks that place a corpus of N
documents in-prompt and ask questions whose difficulty scales with how much of the corpus must be
tracked simultaneously — O(N) retrieval, O(N²) relational tasks (find every contradicting pair),
O(NM)/O(N³) structural ones (cluster everything, find planted triples). Every task has a context
ladder; most reach 1M tokens, the rest stop at a documented source-corpus ceiling.

- **22 tasks** (`ctc_fiqa` … `ctc_scifact`) with per-rung variants (`ctc_nq:r2k` … `:r1m`)
- **41 suites** along two orthogonal axes. By *context length*: `ctc:figure` (the 2k–32k grid, 108
  runs), `ctc:xlong` (>32k, 69 runs), per-rung columns (`ctc:r128k`). By *corpus-tracking demand*,
  the axis the suite is named for: `ctc:low` (the 11 O(N) rows, 102 runs) and `ctc:high` (the 11
  O(N²)/O(NM)/O(N³) rows, 75 runs), which compose with the length axis as `ctc:high:figure` (53)
  and friends. Plus per-task ladders (`ctc:oolong`) and `ctc` for all 177 combinations —
  DISPLAY_ONLY aggregation throughout, since the metrics are heterogeneous by design
- **Data**: HF dataset `PrasannSinghal/ctc-suite-eval` (parquet; one config per task, one split
  per rung; currently private — see Access below). `CTC_SUITE_DATA_ROOT` swaps in a local tree.
- Docs in `src/olmo_eval/evals/tasks/ctc_suite/README.md`

## Design notes for review

- Prompt templates, parsers, metrics, gold-index conventions and stop rules are **vendored
  byte-faithful** under `_vendor/` from the `ctc` package (OLMo-core branch `prasann/ctc`, last
  re-vendored 2026-09-08 at `b7ee85a93`), where
  they are golden-fixture-tested against the implementation that produced the suite's published
  numbers. The vendor tree is ruff-excluded so it stays diffable against upstream; changes flow
  upstream-first, then re-vendor.
- Scoring mirrors the reference runner call-for-call. Per-output metadata records `ctc_parse_ok`
  so parse-rate regressions are distinguishable from capability drops.
- Sub-500-example rungs (everything ≥256k = 125 seeded; scifact 300, obliq 126) carry eval_size
  flags on the roster and in the dataset card.

## Validation

- 22/22 tasks load, format, and grade real rung rows end-to-end; a gold-rendered answer scores
  exactly 1.000 on every task, verified up to real 1M-token examples
- Gold-index conventions pinned by test (`tests/evals/tasks/test_ctc_suite.py`, CI-safe: no
  network, synthetic fixtures)
- Full every-config × every-split `datasets` load sweep run against the Hub copy
- `olmo-eval run -m mock -t ctc:r2k --dry-run` and friends verified
- Rebased onto upstream `main` 2026-09-08 (22 upstream commits, clean merge, no API breakage —
  the upstream changes touching shared surfaces are all additive: `OutputScoreAggregation.MEAN`,
  `MacroSubsetAccuracyMetric`, judge fields on `TaskConfig`, failure accounting on
  `StoredTaskResult`). Full `tests/evals/` suite green (1037 passed); the class split is pinned by
  two new tests; a live 3-instance `ctc_strmatch:r2k` run against the public HF dataset scores
  end-to-end.

## Access

The dataset is **public** (collection:
https://huggingface.co/collections/PrasannSinghal/ctc-suite-long-context-evals-to-1m-tokens-6a7f742925d8d5b6f526f8ba)
— anyone can run the suite with no token. Gold answers are included; the card asks that it be
excluded from pretraining corpora.

## Open questions for maintainers

1. Naming: happy to rename the task prefix / suite names to house conventions.
2. Dataset hosting: keep under my account vs. transfer to allenai — either works with the code
   as-is.
3. The `_vendor/` pattern vs. packaging `ctc` as a pip dependency — vendoring was chosen so the
   branch is self-contained and reviewable, but if you prefer a dependency we can publish the
   package.
