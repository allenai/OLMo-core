# Ladder Run Tracker

Last updated: 2026-06-25 18:45 UTC.

This table is a scan-friendly status matrix for planned ladder cells. It is separate from `RUNS.md` (chronological launch/status log) and `PLOTTED_RESULTS.md` (finished-only plotted rows and losses).

Main experiment categories: baseline, dense schedule, expert granularity, Qwen-like, shared expert, and total sparsity. Rows marked diagnostic are tracked for context but are not part of the main full-grid completion target.

Legend: `done` = at least one finished/plotted run exists; `run` = currently running in Beaker; `queued` = created/scheduled but not started; `todo` = planned/not started; `hold` = intentionally not prioritized yet.

## Current Full-Grid Gaps

| Experiment | Remaining not-yet-queued / not-started cells | Notes |
| --- | --- | --- |
| Total sparsity | 1.2B Cx1/2/4/8 for high total 96E/top4 and huge total 192E/top4 | 275M, 480M, and 810M are done. |
| Dense schedule | None beyond queued/running 810M/1.2B promotions | 480M promoted grid is done. 810M is mostly done with Cx8 tails still running; 1.2B dense0 has started and dense2/dense4 are partly queued. |
| Shared expert | 1.2B Cx8 queued/created | 480M, 810M, and 1.2B Cx1/2/4 are Beaker-finalized with exit code 0. The 2026-06-23 duplicate 480M relaunches were eval-only resumes; duplicate Cx8 was stopped before start. |
| Qwen-like | None beyond currently running 1.2B tail jobs | 810M is done. Active-matched 1.2B Cx1/Cx2/Cx4 are done; remaining 1.2B Cx8 and true-3D tail jobs are running. |
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
| Dense schedule | dense0 + shared | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2/4, run Cx8 | done Cx1, run Cx2/4/8 | 810M Cx4 and 1.2B Cx1 finished since the previous tracker update. |
| Dense schedule | dense2 + shared | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2/4, run Cx8 | run Cx1/2, queued Cx4/8 | 810M Cx2/Cx4 finished since the previous tracker update. |
| Dense schedule | dense4 + shared | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2/4, run Cx8 | queued Cx1/2/4/8 | 810M Cx1/Cx2/Cx4 finished since the previous tracker update. |
| Qwen3-like | active matched 4.5d | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2/4, run Cx8 | 1.2B Cx4 finished and is now plotted. |
| Qwen3-like | true 3.0d + depth | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1, run Cx2/4/8 | 1.2B Cx2 latest attempt is running after requeue; Cx4/Cx8 also running. |

## Active / Queued Beaker Surface

Bounded status pass on 2026-06-25 18:45 UTC checked only runs that were previously running, queued, or created in this table / `RUNS.md`; it did not scan the full historical W&B/Beaker surface.

| Run(s) | State | Latest timestamp UTC | Beaker | Notes |
| --- | --- | --- | --- | --- |
| `q3-1p2b-cx4-q3am128e8k-lr3e-4-r1` | done | finalized 2026-06-25 18:23 | https://beaker.org/ex/01KVJ32TJEXX4XR9FY0QN8PA0T | Active-matched 1.2B Cx4 finished cleanly and is now plotted. |
| `q3-1p2b-cx2-q3td128e8k-lr6e-4-r1` | run | latest attempt started 2026-06-25 13:12 | https://beaker.org/ex/01KVJ33YG7A5DKD5EPZDTSFRPR | Requeued true-3D 1.2B Cx2; not plotted yet. |
| `q3-1p2b-cx4-q3td128e8k-lr3e-4-r1` | run | started 2026-06-24 19:40 | https://beaker.org/ex/01KVJ34BFBN49YH2AKVSHS0GW5 | True-3D 1.2B Cx4. |
| `q3-1p2b-cx8-q3am128e8k-lr4e-4-r1` | run | started 2026-06-24 04:05 | https://beaker.org/ex/01KVJ4GXHKR0DP3PXHPR5ZZ6GB | Corrected active-matched 1.2B Cx8 4e-4 job. |
| `q3-1p2b-cx8-q3td128e8k-lr4e-4-r1` | run | latest attempt started 2026-06-25 18:25 | https://beaker.org/ex/01KVJ4H8PTJDJCGHHFRB8CD3GP | Corrected true-3D 1.2B Cx8 4e-4 job. |
| `ds-810m-cx4-ds0-sh-lr4e-4-r1` | done | finalized 2026-06-25 05:14 | https://beaker.org/ex/01KVV1WC9YCBY0PFA78QM1E12P | Plotted. |
| `ds-810m-cx2-ds2-sh-lr5.6e-4-r1` | done | finalized 2026-06-24 22:57 | https://beaker.org/ex/01KVV1Z3EJ66SEPY2BB4A3ZH59 | Plotted. |
| `ds-810m-cx4-ds2-sh-lr4e-4-r1` | done | finalized 2026-06-25 07:15 | https://beaker.org/ex/01KVV1ZW8RSN4EGZEN54MA18VN | Plotted. |
| `ds-810m-cx1-ds4-sh-lr6e-4-r1` | done | finalized 2026-06-24 21:23 | https://beaker.org/ex/01KVV21W5MQFCX9R6NWC7FCARD | Plotted. |
| `ds-810m-cx2-ds4-sh-lr5.6e-4-r1` | done | finalized 2026-06-25 04:39 | https://beaker.org/ex/01KVV22W9JRH1FFKVVQAYRZV8F | Plotted. |
| `ds-810m-cx4-ds4-sh-lr4e-4-r1` | done | finalized 2026-06-25 17:35 | https://beaker.org/ex/01KVV23RDA44XBKW3D59DBYYEX | Plotted. |
| `ds-810m-cx8-ds{0,2,4}-sh-lr4e-4-r1` | run | started 2026-06-24 08:34 through 21:24 | see `RUNS.md` 2026-06-23 dense launch section | Remaining 810M dense-schedule tail. |
| `ds-1p2b-cx1-ds0-sh-lr4e-4-r1` | done | finalized 2026-06-25 15:12 | https://beaker.org/ex/01KVV26CWEGWR09P5JBVR1QX24 | Plotted. |
| `ds-1p2b-cx{2,4,8}-ds0-sh-*` | run | started 2026-06-25 04:39 through 07:16 | see `RUNS.md` 2026-06-23 dense launch section | dense0 1.2B tail running. |
| `ds-1p2b-cx{1,2}-ds2-sh-*` | run | started 2026-06-25 16:51 through 17:36 | see `RUNS.md` 2026-06-23 dense launch section | dense2 Cx1/Cx2 running. |
| `ds-1p2b-cx{4,8}-ds2-sh-*`, `ds-1p2b-cx{1,2,4,8}-ds4-sh-*` | queued | created 2026-06-23 20:20 through 20:24 | see `RUNS.md` 2026-06-23 dense launch section | Waiting for Titan capacity. |
| `se-1p2b-cx8-se0m9-lr4e-4-r1` | queued | created 2026-06-23 20:24 | https://beaker.org/ex/01KVV2FTHMVKP4ARF5B2A86DN5 | Remaining shared-expert 1.2B Cx8 on Titan urgent, compile-on. |

## Tracking Hygiene

Before launching any promoted run, check all three evidence sources for the exact semantic run name and checkpoint save folder:

1. `RUNS.md` for prior launch records and Beaker IDs.
2. Beaker/W&B finished state for the prior semantic run name.
3. Weka checkpoint folder for final-looking `step*` directories.

Launchers should eventually refuse to submit when the target save folder already contains a final-looking checkpoint unless an explicit override such as `ALLOW_RESUME_FINISHED=1` is set. The tracker should be regenerated from Beaker/W&B/checkpoint evidence rather than manually inferred from the chronological launch log.
