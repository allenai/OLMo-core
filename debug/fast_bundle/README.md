# fast_bundle

Evidence for the fast (shared-corpus) eval bundle — see `records/running-ctc-evals.md` §fast.

- `measure_reuse.py` — measured shared-prefix fraction per task/rung/query-position (the table in
  the record: multiplexed tasks save ~nothing under `both`, ~7–8× under `after`); no GPU needed.
- `check_invariants.py` — a built fast rung's two load-bearing properties: byte-identical shared
  prefix within a corpus group, and gold structure surviving the rebuild.
- `measure_topic_recovery.py` — whether outlier's planted-prefix construction scales to the
  ultra-long rungs (per-topic chunk supply).
- `out/` (gitignored) — built rungs.
