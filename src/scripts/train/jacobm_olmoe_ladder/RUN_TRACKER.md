# Ladder Run Tracker

Last updated: 2026-06-29 20:30 UTC.

This table is a scan-friendly status matrix for planned ladder cells. It is separate from `RUNS.md` (chronological launch/status log) and `PLOTTED_RESULTS.md` (finished-only plotted rows and losses).

Main experiment categories: baseline, dense schedule, expert granularity, Qwen-like, shared expert, and total sparsity. Rows marked diagnostic are tracked for context but are not part of the main full-grid completion target.

Legend: `done` = at least one finished/plotted run exists; `run` = currently running in Beaker; `queued` = created/scheduled but not started; `todo` = planned/not started; `hold` = intentionally not prioritized yet.

## Current Full-Grid Gaps

| Experiment | Remaining not-yet-queued / not-started cells | Notes |
| --- | --- | --- |
| Total sparsity | 1.2B Cx1/2/4/8 for high total 96E/top4 and huge total 192E/top4 | 275M, 480M, and 810M are done. |
| Dense schedule | None beyond running/finished-unplotted 1.2B tail jobs | 480M and 810M are plotted. Several 1.2B dense jobs finished; three 1.2B dense histories are currently blocked by W&B `CommError`. |
| Shared expert | 1.2B Cx8 queued/created | 480M, 810M, and 1.2B Cx1/2/4 are Beaker-finalized with exit code 0. The 2026-06-23 duplicate 480M relaunches were eval-only resumes; duplicate Cx8 was stopped before start. |
| Qwen-like | None beyond restarted true-3D 1.2B Cx8 retry | Active-matched 1.2B Cx8 finished and is plotted. True-3D 1.2B Cx8 was restarted after an exit-1/cancelled attempt and has a fresh running W&B attempt. |
| Expert granularity | None for main coarse/fine grid | Diagnostic 192E/384E remains intentionally limited to 275M Cx1. |
| Baseline | None for Cx1/2/4/8 main grid | Current grid complete. |

## Status Matrix

| Experiment | Variant / comparison | 275M | 480M | 810M | 1.2B | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 48E/top4 | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2/4/8 | Finished-only plots include the canonical Cx8 same-global-batch 4e-4 point. |
| Expert granularity | coarse 24E/top2 | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2/4/8 | Main promoted ladder complete. |
| Expert granularity | fine 96E/top8 | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2/4/8 | Main promoted ladder complete; 1.2B Cx8 finished 2026-06-22. |
| Expert granularity | diagnostic 192E/384E | done Cx1 only | hold | hold | hold | Diagnostic only; intentionally not part of current full ladder. |
| Total sparsity | high total 96E/top4 | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2/4/8 | todo Cx1/2/4/8 | 810M promoted wave complete. |
| Total sparsity | huge total 192E/top4 | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2/4/8 | todo Cx1/2/4/8 | 810M Cx4/Cx8 replacements finished since last status. |
| Shared expert | no shared, routed 9/8 d | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2/4, queued Cx8 | 480M/810M/1.2B Cx1-4 finalized cleanly; 1.2B Cx8 remains queued. Shared plotter 480M name parsing was fixed 2026-06-24 after the status audit. |
| Dense schedule | dense0 + shared | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2, finished-unplotted Cx4, finished-unplotted Cx8 | 810M Cx8 is now plotted. 1.2B Cx4/Cx8 dense0 are Beaker-finished but blocked by W&B `CommError`. |
| Dense schedule | dense2 + shared | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2, finished-unplotted Cx4, run Cx8 | 810M Cx8 is now plotted. 1.2B Cx4 dense2 is blocked by W&B `CommError`. |
| Dense schedule | dense4 + shared | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2, finished-unplotted Cx4, run Cx8 | 1.2B Cx2 dense4 is plotted; Cx4 is Beaker-finished but blocked by W&B `CommError`. |
| Qwen3-like | active matched 4.5d | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2/4/8 | Main active-matched Qwen-like ladder is plotted through Cx8. |
| Qwen3-like | true 3.0d + depth | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2/4, run Cx8 retry | Cx8 has a fresh running W&B attempt after earlier failed/crashed attempts. |

## Active / Queued Beaker Surface

Bounded status pass on 2026-06-29 20:30 UTC checked only runs that were previously running, queued, created, or finished-unplotted in this table / `RUNS.md`; it did not scan the full historical W&B/Beaker surface.

| Run(s) | State | Latest timestamp UTC | Beaker | Notes |
| --- | --- | --- | --- | --- |
| `q3-1p2b-cx8-q3am128e8k-lr4e-4-r1` | done | finalized 2026-06-28 00:43 | https://beaker.org/ex/01KVJ4GXHKR0DP3PXHPR5ZZ6GB | Active-matched 1.2B Cx8 finished cleanly and is now plotted. |
| `q3-1p2b-cx8-q3td128e8k-lr4e-4-r1` | run | fresh W&B attempt created 2026-06-29 18:42 | https://beaker.org/ex/01KVJ4H8PTJDJCGHHFRB8CD3GP | Restarted in-place; latest W&B run is `r96ox1ij`; not plotted until finished. |
| `ds-810m-cx8-ds0-sh-lr4e-4-r1` | done | finalized 2026-06-25 22:54 | https://beaker.org/ex/01KVV1X7T3RPAX3B0SAK4HTKT8 | Full W&B history now cached; plotted. |
| `ds-810m-cx8-ds2-sh-lr4e-4-r1` | done | finalized 2026-06-26 04:28 | https://beaker.org/ex/01KVV20XZKCAXGRMTB9MDVFAWN | Full W&B history now cached; plotted. |
| `ds-810m-cx8-ds4-sh-lr4e-4-r1` | done | finalized 2026-06-26 10:04 | https://beaker.org/ex/01KVV24VVNWDCGF6JSFKXYJ024 | Full W&B history now cached; plotted. |
| `ds-1p2b-cx4-ds0-sh-lr3e-4-r1` | done | finalized 2026-06-27 08:05 | https://beaker.org/ex/01KVV28EST9EFM4ZF43BP2FN15 | Full W&B history now cached; plotted. |
| `ds-1p2b-cx8-ds0-sh-lr4e-4-r1` | finished-unplotted | finalized 2026-06-28 17:45 | https://beaker.org/ex/01KVV29F7ZM9RW8DWF073QYYGF | W&B full-history fetch still returns `CommError` as of 2026-06-29 20:30; semantic run lookup confirms `qpp6fidz` is the matching finished run. |
| `ds-1p2b-cx4-ds2-sh-lr3e-4-r1` | finished-unplotted | finalized 2026-06-28 01:44 | https://beaker.org/ex/01KVV2CCDP7P9C9HM6RYAMM2M8 | W&B full-history fetch still returns `CommError` as of 2026-06-29 20:30; semantic run lookup confirms `rn5yr28o` is the matching finished run. |
| `ds-1p2b-cx8-ds2-sh-lr4e-4-r1` | run | started 2026-06-26 04:29 | https://beaker.org/ex/01KVV2CVAH6ZYWMRYC8TRDS4DJ | dense2 1.2B Cx8. |
| `ds-1p2b-cx2-ds4-sh-lr6e-4-r1` | done | finalized 2026-06-27 10:53 | https://beaker.org/ex/01KVV2DYRCB5ZDNTZHBD0MXGNX | Full W&B history cached; plotted. |
| `ds-1p2b-cx4-ds4-sh-lr3e-4-r1` | finished-unplotted | finalized 2026-06-28 10:19 | https://beaker.org/ex/01KVV2EE8YK7B0MF0EFJ3P9YCZ | W&B full-history fetch still returns `CommError` as of 2026-06-29 20:30; semantic run lookup confirms `2i9wpg3j` is the matching finished run. |
| `ds-1p2b-cx8-ds4-sh-lr4e-4-r1` | run | started 2026-06-26 21:15 | https://beaker.org/ex/01KVV2F1R6M48R02BKM9RVJZH0 | dense4 1.2B Cx8. |
| `se-1p2b-cx8-se0m9-lr4e-4-r1` | run | started 2026-06-26 23:55 | https://beaker.org/ex/01KVV2FTHMVKP4ARF5B2A86DN5 | Shared-expert 1.2B Cx8 on Titan urgent, compile-on. |

## Known Plotting Issues

### Dense Schedule W&B History Fetches

On 2026-06-29, the previously blocked full W&B `scan_history` calls for `5x1zju17`, `abbmdfx0`, `13sr2oht`, and `4x0anaih` succeeded and those runs are now in the canonical dense-schedule plots. The sampled fallback values below are retained only as a record of the temporary diagnostic path; do not mix sampled fallback values into canonical plots unless we explicitly decide to change plotting policy.

Current unresolved dense-schedule history fetches:

| Run ID | Cell | Beaker state | Issue | Note |
| --- | --- | --- | --- | --- |
| `qpp6fidz` | 1.2B Cx8 dense0 | finished | W&B `CommError` during full-history fetch | Retried 2026-06-29 20:30; semantic display-name lookup finds no newer replacement run. |
| `rn5yr28o` | 1.2B Cx4 dense2 | finished | W&B `CommError` during full-history fetch | Retried 2026-06-29 20:30; semantic display-name lookup finds no newer replacement run. |
| `2i9wpg3j` | 1.2B Cx4 dense4 | finished | W&B `CommError` during full-history fetch | Retried 2026-06-29 20:30; semantic display-name lookup finds no newer replacement run; do not plot until exact history is cached. |

Resolved sampled diagnostics from 2026-06-28:

| Run ID | Cell | sampled avg250M | canonical avg250M after full cache | Note |
| --- | --- | ---: | ---: | --- |
| `5x1zju17` | 810M Cx8 dense0 | 2.1860 | 2.1716 | Full history cached 2026-06-29. |
| `abbmdfx0` | 810M Cx8 dense2 | 2.1889 | 2.1741 | Full history cached 2026-06-29. |
| `13sr2oht` | 810M Cx8 dense4 | 2.1918 | 2.1825 | Full history cached 2026-06-29. |
| `4x0anaih` | 1.2B Cx4 dense0 | 2.1510 | 2.1495 | Full history cached 2026-06-29. |

## Tracking Hygiene

Before launching any promoted run, check all three evidence sources for the exact semantic run name and checkpoint save folder:

1. `RUNS.md` for prior launch records and Beaker IDs.
2. Beaker/W&B finished state for the prior semantic run name.
3. Weka checkpoint folder for final-looking `step*` directories.

Launchers should eventually refuse to submit when the target save folder already contains a final-looking checkpoint unless an explicit override such as `ALLOW_RESUME_FINISHED=1` is set. The tracker should be regenerated from Beaker/W&B/checkpoint evidence rather than manually inferred from the chronological launch log.
