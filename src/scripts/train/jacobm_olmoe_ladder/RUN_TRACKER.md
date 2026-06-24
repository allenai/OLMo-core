# Ladder Run Tracker

Last updated: 2026-06-24 21:25 UTC.

This table is a scan-friendly status matrix for planned ladder cells. It is separate from `RUNS.md` (chronological launch/status log) and `PLOTTED_RESULTS.md` (finished-only plotted rows and losses).

Main experiment categories: baseline, dense schedule, expert granularity, Qwen-like, shared expert, and total sparsity. Rows marked diagnostic are tracked for context but are not part of the main full-grid completion target.

Legend: `done` = at least one finished/plotted run exists; `run` = currently running in Beaker; `queued` = created/scheduled but not started; `todo` = planned/not started; `hold` = intentionally not prioritized yet.

## Current Full-Grid Gaps

| Experiment | Remaining not-yet-queued / not-started cells | Notes |
| --- | --- | --- |
| Total sparsity | 1.2B Cx1/2/4/8 for high total 96E/top4 and huge total 192E/top4 | 275M, 480M, and 810M are done. |
| Dense schedule | None beyond queued/running 810M/1.2B promotions | 480M promoted grid is done. 810M is partly done/running/queued; 1.2B is still created/queued on Titan urgent, compile-on. |
| Shared expert | 1.2B Cx8 queued/created | 480M, 810M, and 1.2B Cx1/2/4 are Beaker-finalized with exit code 0. The 2026-06-23 duplicate 480M relaunches were eval-only resumes; duplicate Cx8 was stopped before start. |
| Qwen-like | None beyond currently running 1.2B tail jobs | 810M is done. Active-matched 1.2B Cx1/Cx2 are done; true-3D Cx2 was requeued and is running after repeated failed/preempted attempts. |
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
| Dense schedule | dense0 + shared | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2, run Cx4/8 | queued Cx1/2/4/8 | 480M done; 810M Cx1/Cx2 plotted; 810M Cx4/Cx8 still running. |
| Dense schedule | dense2 + shared | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1, run Cx2/4/8 | queued Cx1/2/4/8 | 480M done; 810M Cx1 plotted; 810M Cx2/Cx4/Cx8 still running. |
| Dense schedule | dense4 + shared | done Cx1/2/4/8 | done Cx1/2/4/8 | run Cx1/2/4, queued Cx8 | queued Cx1/2/4/8 | 480M done; remaining 810M/1.2B tail is still draining. |
| Qwen3-like | active matched 4.5d | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2, run Cx4/8 | 810M Cx8 and 1.2B Cx2 finished since the previous tracker update. |
| Qwen3-like | true 3.0d + depth | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1, run Cx2/4/8 | 1.2B Cx2 failed repeatedly but was requeued on 2026-06-24 and is running in latest attempt. |

## Active / Queued Beaker Surface

Bounded status pass on 2026-06-24 21:25 UTC checked only runs that were previously running, queued, or created in this table / `RUNS.md`; it did not scan the full historical W&B/Beaker surface.

| Run(s) | State | Latest timestamp UTC | Beaker | Notes |
| --- | --- | --- | --- | --- |
| `q3-810m-cx8-q3am128e8k-lr4e-4-r1` | done | finalized 2026-06-23 21:00 | https://beaker.org/ex/01KVFDNGV9TDAP885HG5MD68FB | Active-matched Qwen-like 810M Cx8 finished cleanly and is now plotted. |
| `q3-1p2b-cx4-q3am128e8k-lr3e-4-r1` | run | started 2026-06-24 19:37 | https://beaker.org/ex/01KVJ32TJEXX4XR9FY0QN8PA0T | Active-matched 1.2B Cx4. |
| `q3-1p2b-cx1-q3td128e8k-lr4e-4-r1` | done | finalized 2026-06-24 04:04 | https://beaker.org/ex/01KVJ33HPRPCQ6ERM46864M1Z3 | True-3D 1.2B Cx1 finished cleanly and is now plotted. |
| `q3-1p2b-cx2-q3am128e8k-lr6e-4-r1` | done | finalized 2026-06-24 13:52 | https://beaker.org/ex/01KVJ32E9D5MQ00NNAW4QJTGMW | Active-matched 1.2B Cx2 finished cleanly and is now plotted. |
| `q3-1p2b-cx2-q3td128e8k-lr6e-4-r1` | run | latest attempt started 2026-06-24 21:14 | https://beaker.org/ex/01KVJ33YG7A5DKD5EPZDTSFRPR | Requeued after repeated exit-1 attempts; not plotted yet. |
| `q3-1p2b-cx4-q3td128e8k-lr3e-4-r1` | run | started 2026-06-24 19:40 | https://beaker.org/ex/01KVJ34BFBN49YH2AKVSHS0GW5 | True-3D 1.2B Cx4. |
| `q3-1p2b-cx8-q3am128e8k-lr4e-4-r1` | run | started 2026-06-24 04:05 | https://beaker.org/ex/01KVJ4GXHKR0DP3PXHPR5ZZ6GB | Corrected active-matched 1.2B Cx8 4e-4 job. |
| `q3-1p2b-cx8-q3td128e8k-lr4e-4-r1` | run | started 2026-06-24 19:46 | https://beaker.org/ex/01KVJ4H8PTJDJCGHHFRB8CD3GP | Corrected true-3D 1.2B Cx8 4e-4 job. |
| `ds-480m-cx{1,2,4,8}-ds{0,2,4}-sh-*-r1` | done | finalized 2026-06-23 23:49 through 2026-06-24 09:25 | see `RUNS.md` 2026-06-23 dense launch section | All 12 promoted 480M dense-schedule runs finished cleanly and are now plotted. |
| `ds-810m-cx1-ds0-sh-lr6e-4-r1` | done | finalized 2026-06-24 10:08 | https://beaker.org/ex/01KVV1TQEETCC76CS0R2JNH9YZ | Plotted. |
| `ds-810m-cx2-ds0-sh-lr5.6e-4-r1` | done | finalized 2026-06-24 21:06 | https://beaker.org/ex/01KVV1VH6PBPP31RD0SM58BM46 | Plotted. |
| `ds-810m-cx1-ds2-sh-lr6e-4-r1` | done | finalized 2026-06-24 15:50 | https://beaker.org/ex/01KVV1Y7V48ZG09C34XYHNSRZ2 | Plotted. |
| `ds-810m-cx{4,8}-ds0-sh-*`, `ds-810m-cx{2,4,8}-ds2-sh-*`, `ds-810m-cx{1,2,4}-ds4-sh-*` | run | started 2026-06-24 07:52 through 21:07 | see `RUNS.md` 2026-06-23 dense launch section | Nine 810M dense-schedule jobs are running. |
| `ds-810m-cx8-ds4-sh-lr4e-4-r1` | queued | created 2026-06-23 20:24 | https://beaker.org/ex/01KVV24VVNWDCGF6JSFKXYJ024 | Waiting for Titan capacity. |
| `ds-1p2b-cx{1,2,4,8}-ds{0,2,4}-sh-*-r1` | queued | created 2026-06-23 20:17 through 20:24 | see `RUNS.md` 2026-06-23 dense launch section | All 12 promoted 1.2B dense-schedule jobs are still created/queued. |
| `se-1p2b-cx8-se0m9-lr4e-4-r1` | queued | created 2026-06-23 20:24 | https://beaker.org/ex/01KVV2FTHMVKP4ARF5B2A86DN5 | Remaining shared-expert 1.2B Cx8 on Titan urgent, compile-on. |

## Tracking Hygiene

Before launching any promoted run, check all three evidence sources for the exact semantic run name and checkpoint save folder:

1. `RUNS.md` for prior launch records and Beaker IDs.
2. Beaker/W&B finished state for the prior semantic run name.
3. Weka checkpoint folder for final-looking `step*` directories.

Launchers should eventually refuse to submit when the target save folder already contains a final-looking checkpoint unless an explicit override such as `ALLOW_RESUME_FINISHED=1` is set. The tracker should be regenerated from Beaker/W&B/checkpoint evidence rather than manually inferred from the chronological launch log.
