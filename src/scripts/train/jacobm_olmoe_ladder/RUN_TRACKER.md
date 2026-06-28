# Ladder Run Tracker

Last updated: 2026-06-28 07:55 UTC.

This table is a scan-friendly status matrix for planned ladder cells. It is separate from `RUNS.md` (chronological launch/status log) and `PLOTTED_RESULTS.md` (finished-only plotted rows and losses).

Main experiment categories: baseline, dense schedule, expert granularity, Qwen-like, shared expert, and total sparsity. Rows marked diagnostic are tracked for context but are not part of the main full-grid completion target.

Legend: `done` = at least one finished/plotted run exists; `run` = currently running in Beaker; `queued` = created/scheduled but not started; `todo` = planned/not started; `hold` = intentionally not prioritized yet.

## Current Full-Grid Gaps

| Experiment | Remaining not-yet-queued / not-started cells | Notes |
| --- | --- | --- |
| Total sparsity | 1.2B Cx1/2/4/8 for high total 96E/top4 and huge total 192E/top4 | 275M, 480M, and 810M are done. |
| Dense schedule | None beyond running 1.2B tail jobs | 480M done. 810M jobs are Beaker-finished, but Cx8 histories are not plotted yet because W&B history fetch is hanging. 1.2B is partly done/running. |
| Shared expert | 1.2B Cx8 queued/created | 480M, 810M, and 1.2B Cx1/2/4 are Beaker-finalized with exit code 0. The 2026-06-23 duplicate 480M relaunches were eval-only resumes; duplicate Cx8 was stopped before start. |
| Qwen-like | None beyond currently running 1.2B Cx8 tail jobs | 810M done. 1.2B Cx1/2/4 are done for both variants; Cx8 runs remain running. |
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
| Dense schedule | dense0 + shared | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2/4, finished-unplotted Cx8 | done Cx1/2, finished-unplotted Cx4, run Cx8 | 810M Cx8 and 1.2B Cx4 are Beaker-finished but skipped by plots because W&B history fetch hangs before returning rows. |
| Dense schedule | dense2 + shared | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2/4, finished-unplotted Cx8 | done Cx1/2, run Cx4/8 | 1.2B Cx1/Cx2 now plotted; 810M Cx8 history fetch is currently blocked. |
| Dense schedule | dense4 + shared | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2/4, finished-unplotted Cx8 | done Cx1, run Cx2/4/8 | 1.2B Cx1 now plotted; 810M Cx8 history fetch is currently blocked. |
| Qwen3-like | active matched 4.5d | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2/4, run Cx8 | Cx8 still running. |
| Qwen3-like | true 3.0d + depth | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2/4, run Cx8 | Cx2 and Cx4 finished and are now plotted; Cx8 still running. |

## Active / Queued Beaker Surface

Bounded status pass on 2026-06-28 07:55 UTC checked only runs that were previously running, queued, or created in this table / `RUNS.md`; it did not scan the full historical W&B/Beaker surface.

| Run(s) | State | Latest timestamp UTC | Beaker | Notes |
| --- | --- | --- | --- | --- |
| `q3-1p2b-cx2-q3td128e8k-lr6e-4-r1` | done | finalized 2026-06-26 10:16 | https://beaker.org/ex/01KVJ33YG7A5DKD5EPZDTSFRPR | True-3D 1.2B Cx2 finished cleanly and is now plotted. |
| `q3-1p2b-cx4-q3td128e8k-lr3e-4-r1` | done | finalized 2026-06-27 03:23 | https://beaker.org/ex/01KVJ34BFBN49YH2AKVSHS0GW5 | True-3D 1.2B Cx4 finished cleanly and is now plotted. |
| `q3-1p2b-cx8-q3am128e8k-lr4e-4-r1` | run | latest attempt started 2026-06-27 09:23 | https://beaker.org/ex/01KVJ4GXHKR0DP3PXHPR5ZZ6GB | Active-matched 1.2B Cx8. |
| `q3-1p2b-cx8-q3td128e8k-lr4e-4-r1` | run | latest attempt started 2026-06-27 09:24 | https://beaker.org/ex/01KVJ4H8PTJDJCGHHFRB8CD3GP | True-3D 1.2B Cx8. |
| `ds-810m-cx8-ds0-sh-lr4e-4-r1` | finished-unplotted | finalized 2026-06-25 22:54 | https://beaker.org/ex/01KVV1X7T3RPAX3B0SAK4HTKT8 | W&B history fetch hangs before returning rows for run `5x1zju17`; skipped in cache-safe plot refresh. |
| `ds-810m-cx8-ds2-sh-lr4e-4-r1` | finished-unplotted | finalized 2026-06-26 04:28 | https://beaker.org/ex/01KVV20XZKCAXGRMTB9MDVFAWN | W&B history fetch not attempted after ds0 Cx8 blocked; still needs cache backfill. |
| `ds-810m-cx8-ds4-sh-lr4e-4-r1` | finished-unplotted | finalized 2026-06-26 10:04 | https://beaker.org/ex/01KVV24VVNWDCGF6JSFKXYJ024 | W&B history fetch not attempted after ds0 Cx8 blocked; still needs cache backfill. |
| `ds-1p2b-cx2-ds0-sh-lr6e-4-r1` | done | finalized 2026-06-26 08:39 | https://beaker.org/ex/01KVV27GXZ5DRQMRKWHN5G5BYA | Plotted. |
| `ds-1p2b-cx4-ds0-sh-lr3e-4-r1` | finished-unplotted | finalized 2026-06-27 08:05 | https://beaker.org/ex/01KVV28EST9EFM4ZF43BP2FN15 | W&B history fetch hangs before returning rows for run `4x0anaih`; skipped in cache-safe plot refresh. |
| `ds-1p2b-cx8-ds0-sh-lr4e-4-r1` | run | started 2026-06-25 07:16 | https://beaker.org/ex/01KVV29F7ZM9RW8DWF073QYYGF | dense0 1.2B Cx8. |
| `ds-1p2b-cx1-ds2-sh-lr4e-4-r1` | done | finalized 2026-06-26 08:39 | https://beaker.org/ex/01KVV2AGRGJBFMPVKVN64G5754 | Plotted. |
| `ds-1p2b-cx2-ds2-sh-lr6e-4-r1` | done | finalized 2026-06-26 21:14 | https://beaker.org/ex/01KVV2BE87YJBG7E68XCKRMTH4 | Plotted. |
| `ds-1p2b-cx4-ds2-sh-lr3e-4-r1` | run | started 2026-06-25 22:55 | https://beaker.org/ex/01KVV2CCDP7P9C9HM6RYAMM2M8 | dense2 1.2B Cx4. |
| `ds-1p2b-cx8-ds2-sh-lr4e-4-r1` | run | started 2026-06-26 04:29 | https://beaker.org/ex/01KVV2CVAH6ZYWMRYC8TRDS4DJ | dense2 1.2B Cx8. |
| `ds-1p2b-cx1-ds4-sh-lr4e-4-r1` | done | finalized 2026-06-26 23:54 | https://beaker.org/ex/01KVV2DBHBF9BPT3SS3VH5K71W | Plotted. |
| `ds-1p2b-cx2-ds4-sh-lr6e-4-r1` | run | started 2026-06-26 08:40 | https://beaker.org/ex/01KVV2DYRCB5ZDNTZHBD0MXGNX | dense4 1.2B Cx2. |
| `ds-1p2b-cx4-ds4-sh-lr3e-4-r1` | run | started 2026-06-26 10:04 | https://beaker.org/ex/01KVV2EE8YK7B0MF0EFJ3P9YCZ | dense4 1.2B Cx4. |
| `ds-1p2b-cx8-ds4-sh-lr4e-4-r1` | run | started 2026-06-26 21:15 | https://beaker.org/ex/01KVV2F1R6M48R02BKM9RVJZH0 | dense4 1.2B Cx8. |
| `se-1p2b-cx8-se0m9-lr4e-4-r1` | run | started 2026-06-26 23:55 | https://beaker.org/ex/01KVV2FTHMVKP4ARF5B2A86DN5 | Shared-expert 1.2B Cx8 on Titan urgent, compile-on. |


## Known Plotting Issues

### Dense Schedule W&B History Fetches

On 2026-06-28, full W&B `scan_history` still hung before returning rows for some Beaker-finished dense-schedule runs, so they remain marked `finished-unplotted` rather than being added to canonical plots. Do not mix sampled fallback values into the main plots unless we explicitly decide to change plotting policy; current canonical plots use full-history final-window averages.

Sampled `run.history()` did return rough fallback values, useful only as diagnostics until full histories can be cached:

| Run ID | Cell | sampled avg250M | Note |
| --- | --- | ---: | --- |
| `5x1zju17` | 810M Cx8 dense0 | 2.1860 | Full `scan_history` hangs; retry later. |
| `abbmdfx0` | 810M Cx8 dense2 | 2.1889 | Full `scan_history` hangs; retry later. |
| `13sr2oht` | 810M Cx8 dense4 | 2.1918 | Sampled fallback works; full cache still needed. |
| `4x0anaih` | 1.2B Cx4 dense0 | 2.1510 | Full `scan_history` hangs; retry later. |

## Tracking Hygiene

Before launching any promoted run, check all three evidence sources for the exact semantic run name and checkpoint save folder:

1. `RUNS.md` for prior launch records and Beaker IDs.
2. Beaker/W&B finished state for the prior semantic run name.
3. Weka checkpoint folder for final-looking `step*` directories.

Launchers should eventually refuse to submit when the target save folder already contains a final-looking checkpoint unless an explicit override such as `ALLOW_RESUME_FINISHED=1` is set. The tracker should be regenerated from Beaker/W&B/checkpoint evidence rather than manually inferred from the chronological launch log.
