# CTC dense-vs-chunked: setting verification of the flagged numbers (2026-07-23)

The user flagged several grid numbers as "weird / too low / disagreeing with the complexity
hypothesis" and asked to confirm each setting is *completely working* before trusting it. Read-only
audits (data + generations + grading code + the `TimeToPayAttention` paper). Verdicts:

## ✅ qdmatch_hpqa ("hpqa") dense ~0.99-flat — REAL, no bug
Genuine O(M·N) sparse-matching (M queries + N docs share one index; k=3 real HotpotQA-bridge pairs,
haystack grows 9→178 across 2k→32k while k stays 3). Grading sound (`_eval_qdmatch`/`pair_metrics`,
set-based exact `(qid,did)` F1, parse_rate=1.0). Generations non-degenerate (byte-exact at 2k;
sensible near-misses at 32k). dense≫chunked (0.99 vs 0.65→0.33) is the CTC thesis. hpqa is *easy for
dense* because bridge entities give strong lexical cues — a task property. **No fix.** (scifact-dense
separately confirmed healthy post-retrain: 0.963→0.879, eval_size=300.)

## ⚠️ cycle / groups4 — exploitable shortcut; NOT a real O(N³) test
`generate_cycle_data.py` draws the K=1 length-3 cycle from a name pool **disjoint** from the
distractor DAG entities, so the 3 cycle names occur exactly 2× while distractor-entity frequency grows
with N. The cycle becomes *more* distinguishable as N grows via a pure frequency heuristic → functionally
O(N) needle-in-haystack, which is why cycle stays ~1.0. `generate_groups4_data.py` has the same pattern
(every distractor value > X from all others → sort-and-scan). **Also: the paper never tests O(N³)** —
Fig 4 is only O(N)/O(N·M)/O(N²); our O(N³) label is our own extrapolation. FIX: embed the gold cycle in
the same connectivity/frequency distribution as distractors (comparable entity degree), and/or scale
`num_cycles`/near-miss almost-cycles with N. Requires generator change + eval (and train) data rebuild.

## ⚠️ grouping "too low" — generator confound, not a broken eval
`sample_k_for_level` sets group count as a fraction of n_docs per level; coarse level L0 needs `k`
distinct OpenAlex top-level fields (only ~19–26 exist), so at high N the required k collides with that
ceiling and `build_example` returns None → the level mix drifts finer as N grows (L0 share 57%→0% over
2k→32k; mean gold cluster 2.27→1.16; singletons 59%→85%). The 0.44→0.01 curve conflates N-scaling with
granularity-hardening → overstates collapse. Model fails by over-merging (precision collapse, recall
~1.0), not attention-range — which is why dense==chunked. FIX: hold the level/k-density distribution
fixed across rungs (or stratify + report per level). Generator change + data rebuild.

## contradiction chunked — anneal recipe is CORRECT (user-verified); other factors remain
An earlier hypothesis that the `chunked-mix` anneal (`mix_start_p=0.80 → mix_end_p=0.0`,
`train_ctc_suite.py:759-760`) was the culprit is **REFUTED**: the user has verified that annealing works
*better* than no-annealing, so the anneal-to-0 recipe is correct and is NOT depressing the grid. The
+0.019 pure→mix delta cited from the 0.8B numbers came from a weak smoke run evaluated in *dense* mode
(`ctc-smoke-contra-3ep`, provenance `evaluator_variant: dense`), not a like-for-like chunked comparison —
so it does not indicate a recipe problem. The mask-mix pilot launched to test fixed-p vs anneal was
cancelled once this was clarified.
What DOES stand from the audit: the cross-chunk severing is correct (claims joined by `\n\n`, isolated
per chunk — verified); the contradiction "gap doesn't widen" is partly the *dense* arm degrading faster
than the paper's flat full-attention baseline, plus a **digit-ID artifact at 32k** (doc indices reach 4
digits → near-miss predictions like gold 1127→117 appear at ~19%), which inflates how bad dense looks at
the tail. The missing **32k chunked-mix contradiction rung** is the single most informative point to fill.
`MIX_START_P`/`MIX_END_P` env passthrough was added to `run_ctc_local.sbatch` (harmless; leave the
default anneal in place).

Minor cleanups noted by the audits: relabel the grid's 4B "chunked" row as "chunked-mix" (it IS the
cmix checkpoint); fix the stale `contradiction: \n-separated → one chunk` docstring in
`src/scripts/ctc_eval/lib/chunked_attention.py`; fill the missing 32k chunked-mix contradiction rung.

## Related
oolong is a separate train/eval wrapping mismatch (wrong training source) — see
`debug/ctc_vllm_validation/CHUNK_LEAK_AUDIT.md`; fix in progress (rebuild from synth split-train +
retrain).
