# Ladder Run Tracker

Last updated: 2026-06-23 UTC.

This table is a scan-friendly status matrix for planned ladder cells. It is separate from `RUNS.md` (chronological launch/status log) and `PLOTTED_RESULTS.md` (finished-only plotted rows and losses).

Main experiment categories: baseline, dense schedule, expert granularity, Qwen-like, shared expert, and total sparsity. Rows marked diagnostic are tracked for context but are not part of the main full-grid completion target.

Legend: `done` = at least one finished/plotted run exists; `run` = currently running in Beaker; `queued` = created/scheduled but not started; `todo` = planned/not started; `hold` = intentionally not prioritized yet.

## Current Full-Grid Gaps

| Experiment | Remaining not-yet-queued / not-started cells | Notes |
| --- | --- | --- |
| Total sparsity | 1.2B Cx1/2/4/8 for high total 96E/top4 and huge total 192E/top4 | 275M, 480M, and 810M are done. |
| Dense schedule | 480M, 810M, and 1.2B Cx1/2/4/8 for dense0, dense2, and dense4 | 275M LR searches are done; promotions have not started. |
| Shared expert | 480M Cx1/2/4/8 and 1.2B Cx8 for no-shared routed-9/8d | Shared expert is a main experiment category; remaining cells were held/canceled earlier for queue pressure, not excluded. |
| Qwen-like | None beyond currently running/queued 810M/1.2B tail jobs | Let the active/queued Beaker surface drain unless a run fails. |
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
| Shared expert | no shared, routed 9/8 d | done Cx1/2/4/8 | todo Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2/4, todo Cx8 | Main experiment category; 480M was canceled earlier to avoid flooding and 1.2B Cx8 remains to launch. |
| Dense schedule | dense0 + shared | done Cx1/2/4/8 | todo Cx1/2/4/8 | todo Cx1/2/4/8 | todo Cx1/2/4/8 | Only 275M LR search finished so far. |
| Dense schedule | dense2 + shared | done Cx1/2/4/8 | todo Cx1/2/4/8 | todo Cx1/2/4/8 | todo Cx1/2/4/8 | Only 275M LR search finished so far. |
| Dense schedule | dense4 + shared | done Cx1/2/4/8 | todo Cx1/2/4/8 | todo Cx1/2/4/8 | todo Cx1/2/4/8 | Only 275M LR search finished so far. |
| Qwen3-like | active matched 4.5d | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2/4, run Cx8 | done Cx1, queued Cx2/Cx8, run Cx4 | 1.2B Cx2 has had low-priority preemptions; latest attempt is created. |
| Qwen3-like | true 3.0d + depth | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2/4/8 | run Cx1, queued Cx2/4/8 | 810M Cx8 finished 2026-06-23. |

## Active / Queued Beaker Surface

| Run | State | Latest timestamp UTC | Beaker | Notes |
| --- | --- | --- | --- | --- |
| `q3-810m-cx8-q3am128e8k-lr4e-4-r1` | run | started 2026-06-23 19:00 | https://beaker.org/ex/01KVFDNGV9TDAP885HG5MD68FB | Many low-priority preemptions; no successful attempt yet. |
| `q3-1p2b-cx4-q3am128e8k-lr3e-4-r1` | run | started 2026-06-23 17:52 | https://beaker.org/ex/01KVJ32TJEXX4XR9FY0QN8PA0T | Active-matched 1.2B Cx4. |
| `q3-1p2b-cx1-q3td128e8k-lr4e-4-r1` | run | started 2026-06-23 19:00 | https://beaker.org/ex/01KVJ33HPRPCQ6ERM46864M1Z3 | True-3D 1.2B Cx1. |
| `q3-1p2b-cx2-q3am128e8k-lr6e-4-r1` | queued | created 2026-06-23 19:02 | https://beaker.org/ex/01KVJ32E9D5MQ00NNAW4QJTGMW | Active-matched 1.2B Cx2; previous attempts were preempted. |
| `q3-1p2b-cx2-q3td128e8k-lr6e-4-r1` | queued | created 2026-06-23 18:31 | https://beaker.org/ex/01KVJ33YG7A5DKD5EPZDTSFRPR | True-3D 1.2B Cx2. |
| `q3-1p2b-cx4-q3td128e8k-lr3e-4-r1` | queued | created 2026-06-23 18:31 | https://beaker.org/ex/01KVJ34BFBN49YH2AKVSHS0GW5 | True-3D 1.2B Cx4. |
| `q3-1p2b-cx8-q3am128e8k-lr4e-4-r1` | queued | created 2026-06-20 09:06 | https://beaker.org/ex/01KVJ4GXHKR0DP3PXHPR5ZZ6GB | Corrected active-matched 1.2B Cx8 4e-4 job. |
| `q3-1p2b-cx8-q3td128e8k-lr4e-4-r1` | queued | created 2026-06-20 09:07 | https://beaker.org/ex/01KVJ4H8PTJDJCGHHFRB8CD3GP | Corrected true-3D 1.2B Cx8 4e-4 job. |
