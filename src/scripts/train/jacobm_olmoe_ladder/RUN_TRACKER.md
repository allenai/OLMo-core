# Ladder Run Tracker

Last updated: 2026-07-02 03:10 UTC.

This table is a scan-friendly status matrix for planned ladder cells. It is separate from `RUNS.md` (chronological launch/status log) and `PLOTTED_RESULTS.md` (finished-only plotted rows and losses).

Main experiment categories: baseline, dense schedule, expert granularity, integration candidates, Qwen-like, shared expert, and total sparsity. Rows marked diagnostic are tracked for context but are not part of the main full-grid completion target.

Legend: `done` = at least one finished/plotted run exists; `run` = currently running in Beaker; `queued` = created/scheduled but not started; `todo` = planned/not started; `hold` = intentionally not prioritized yet.

## Current Full-Grid Gaps

| Experiment | Remaining not-yet-queued / not-started cells | Notes |
| --- | --- | --- |
| Total sparsity | 1.2B Cx1/2/4/8 for high total 96E/top4 and huge total 192E/top4 | 275M, 480M, and 810M are done. |
| Integration candidates | 275M LR grid is plotted for Cx1/2/4/8; extra cold-side wide Cx8 point is queued/running; 480M wide/deep promotions are in flight | 275M wide Cx4 is now bracketed after the `4e-4` cold point; 480M deep Cx1/Cx2 and 480M wide Cx4 finished/plotted; wide/deep points are beating baseline at same LR so far. |
| Dense schedule | None | 480M, 810M, and 1.2B dense jobs are now finished/plotted. |
| Shared expert | None | 480M, 810M, and 1.2B Cx1/2/4/8 are Beaker-finalized and plotted. |
| Qwen-like | None | Active-matched and true-3D Qwen-like grids are finished/plotted through 1.2B Cx8. |
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
| Shared expert | no shared, routed 9/8 d | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2/4/8 | Promoted ladder complete and plotted. Shared plotter 480M name parsing was fixed 2026-06-24 after the status audit. |
| Dense schedule | dense0 + shared | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2/4/8 | 1.2B Cx8 dense0 is plotted after full history eventually cached. |
| Dense schedule | dense2 + shared | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2/4/8 | Promoted ladder complete and plotted. |
| Dense schedule | dense4 + shared | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2/4/8 | Promoted ladder complete and plotted; some 1.2B dense4 rows use exact tail history. |
| Qwen3-like | active matched 4.5d | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2/4/8 | Main active-matched Qwen-like ladder is plotted through Cx8. |
| Qwen3-like | true 3.0d + depth | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2/4/8 | done Cx1/2/4/8 | Main true-3D Qwen-like ladder is plotted through Cx8 after the in-place restart. |
| Integration candidates | wide 256E/top8 + shared + dense1 | Cx1/Cx2/Cx4/Cx8 done/plotted and bracketed | Cx1/Cx2/Cx4/Cx8 done/plotted | queued/created | todo | 480M wide Cx1/Cx2/Cx4/Cx8 are beating same-LR baseline; 810M Cx1/Cx2/Cx4/Cx8 launched on Titan urgent. |
| Integration candidates | deep 256E/top8 + shared + dense1 | Cx1/2/4/8 done/plotted | Cx1/Cx2 done/plotted; Cx4/Cx8 queued/running | queued/created | todo | 480M deep Cx1/Cx2 are beating wide and baseline at same LR so far; 810M Cx1/Cx2/Cx4/Cx8 launched on Titan urgent. |

## Active / Queued Beaker Surface

Bounded status pass on 2026-07-01 05:00 UTC checked only runs that were previously running, queued, created, or finished-unplotted in this table / `RUNS.md`; it did not scan the full historical W&B/Beaker surface.

| Run(s) | State | Latest timestamp UTC | Beaker | Notes |
| --- | --- | --- | --- | --- |
| `int-810m-cx{1,2,4,8}-intw256e8k-baseline-LR-r1` | queued/created | created 2026-07-02 06:18-06:19 | https://beaker.org/ex/01KWGQP0NGMXDEN2PBRGAXJZ7R | 810M wide integration promoted single points on Titan urgent, compile-on. Cx1 `6e-4`, Cx2 `5.6e-4`, Cx4/Cx8 `4e-4`; GBS seq 32/48/64/96; GPUs 8/8/8/8; MB 4/2/4/4. Beaker IDs: `01KWGQP0...`, `01KWGQPD...`, `01KWGQPR...`, `01KWGQQ4...`. |
| `int-810m-cx{1,2,4,8}-intd256e8k-baseline-LR-r1` | queued/created | created 2026-07-02 06:19-06:20 | https://beaker.org/ex/01KWGQQVD0TE5ZY5K5T05GEYAK | 810M deep integration promoted single points on Titan urgent, compile-on. Cx1 `6e-4`, Cx2 `5.6e-4`, Cx4/Cx8 `4e-4`; GBS seq 32/48/64/96; GPUs 8/8/8/8; MB 4/2/4/4. Beaker IDs: `01KWGQQV...`, `01KWGQR7...`, `01KWGQRJ...`, `01KWGQRX...`. |
| `int-480m-cx{1,2,4,8}-intw256e8k-baseline-LR-r1` | queued/created | created 2026-07-01 05:37-05:38 | https://beaker.org/ex/01KWE2XDE9NATMWCWKAH9X29JT | 480M wide integration promoted single points on Titan urgent, compile-on. Cx1 `1.2e-3`, Cx2 `9e-4`, Cx4/Cx8 `8e-4`; GBS seq 32/48/64/96; GPUs 4/4/4/8; MB 4 throughout. Beaker IDs: `01KWE2XDE...`, `01KWE2XSK...`, `01KWE2Y61...`, `01KWE2YHF...`. |
| `int-275m-cx1-intw256e8k-lr8e-4-r1` | run | restarted attempt started 2026-07-01 02:47 | https://beaker.org/ex/01KWDDW61H689812K3DWHWH97W | Original attempt exited 1 at 2026-06-30 23:51; user restarted in-place and fresh attempt is running. |
| `int-275m-cx1-intw256e8k-lr1.6e-3-r1` | done | finalized 2026-07-01 02:39 | https://beaker.org/ex/01KWDDWKR6E5ZGKGE0114WM851 | Wide integration Cx1 mid LR finished cleanly and is plotted from tail history. |
| `int-275m-cx1-intw256e8k-lr3.2e-3-r1` | done | finalized 2026-07-01 02:28 | https://beaker.org/ex/01KWDDWZ15ET9GRVJB2NT7W6FZ | Wide integration Cx1 hot LR finished cleanly and is plotted from tail history. |
| `int-275m-cx2-intw256e8k-lr{8e-4,1.6e-3,3.2e-3}-r1` | done | finalized 2026-07-01 04:41 | https://beaker.org/ex/01KWDDXAREEBFDZZQ4PK00EBTR | All three wide integration Cx2 jobs finished cleanly and are plotted (`01KWDDXARE...`, `01KWDDXPPS...`, `01KWDDY2PT...`). |
| `int-275m-cx4-intw256e8k-lr8e-4-r1` | done | finalized 2026-07-01 04:50 | https://beaker.org/ex/01KWDDYE49G1366Q1EQFD3S7P5 | Cold wide integration Cx4 finished cleanly and is plotted. |
| `int-275m-cx4-intw256e8k-lr{1.6e-3,3.2e-3}-r1` | run | started 2026-07-01 04:35 | https://beaker.org/ex/01KWDDYSVFBA5PC370YP3YF33C | Wide integration Cx4 mid/hot LRs are running (`01KWDDYSVF...`, `01KWDDZ61...`). |
| `int-275m-cx8-intw256e8k-lr{8e-4,1.6e-3,3.2e-3}-r1` | run | started 2026-07-01 04:37-04:39 | https://beaker.org/ex/01KWDDZJ4VKPJEK5Z4M3EW5MM8 | Wide integration Cx8 grid running (`01KWDDZJ4...`, `01KWDDZXH...`, `01KWDE09E...`). |
| `int-275m-cx1-intd256e8k-lr{8e-4,1.6e-3,3.2e-3}-r1` | done | plotted 2026-07-01 09:48 | https://beaker.org/ex/01KWDE0ME50VN6F2YJW0Z3ZVF3 | Deep Cx1 finished and is plotted. Best observed `1.6e-3`; fit `~1.32e-3`, about `0.65x` baseline Cx1 fit. |
| `int-275m-cx{2,4,8}-intd256e8k-lr{8e-4,1.6e-3,3.2e-3}-r1` | done | plotted 2026-07-01 10:17 | https://beaker.org/ex/01KWDE1RNPPST5WDYR5YB7PWMH | Deep Cx2/Cx4/Cx8 grids are finished and plotted. Cx2 best observed `1.6e-3`; fit `~1.47e-3`, about `0.83x` baseline Cx2 fit. |
| `int-275m-cx4-intw256e8k-lr4e-4-r1` | done | plotted 2026-07-02 03:10 | https://beaker.org/ex/01KWFRX0XC1823E3F0NG4VV9F4 | Cold-side wide Cx4 follow-up finished. `4e-4` avg250M `2.5206` vs `8e-4` avg250M `2.5060`, so Cx4 is now bracketed with observed best at `8e-4`. |
| `int-275m-cx8-intw256e8k-lr4e-4-r1` | queued/running | created 2026-07-01 21:21 | https://beaker.org/ex/01KWFRYPK82AC42HWXR0HNRE3G | Remaining cold-side follow-up for unbracketed wide 275M Cx8. GBS seq 96, 8 GPUs, MB4. |
| `int-480m-cx{1,2}-intd256e8k-baseline-LR-r1` | done | plotted 2026-07-02 02:00 | https://beaker.org/ex/01KWF7Z3GFWM8JS31NB0P9M516 | 480M deep Cx1/Cx2 finished. Cx1 `1.2e-3` avg250M `2.5291`; Cx2 `9e-4` avg250M `2.4091`. Both beat wide and same-LR baseline. |
| `int-480m-cx{4,8}-intd256e8k-baseline-LR-r1` | queued/running | checked 2026-07-02 02:00 | https://beaker.org/ex/01KWF804APPWNTTXT0B118G5MB | 480M deep Cx4/Cx8 not yet plotted. |
| `int-480m-cx{1,2}-intw256e8k-baseline-LR-r1` | done | plotted 2026-07-01 10:17 | https://beaker.org/ex/01KWE2XDE9NATMWCWKAH9X29JT | 480M wide Cx1/Cx2 finished. Cx1 `1.2e-3` avg250M `2.5433` vs baseline same-LR `2.5636`; Cx2 `9e-4` avg250M `2.4239` vs baseline same-LR `2.4630`. |
| `int-480m-cx4-intw256e8k-baseline-LR-r1` | done | plotted 2026-07-02 02:00 | https://beaker.org/ex/01KWE2Y61CJEQEDW57MDJEPRDH | 480M wide Cx4 `8e-4` avg250M `2.3300` vs baseline same-LR `2.3788`. |
| `int-480m-cx8-intw256e8k-baseline-LR-r1` | queued/running | checked 2026-07-02 02:00 | https://beaker.org/ex/01KWE2YHFRQHAF56KCK95FPK62 | 480M wide Cx8 not yet plotted. |
| `q3-1p2b-cx8-q3td128e8k-lr4e-4-r1` | done | finalized 2026-06-30 17:37 | https://beaker.org/ex/01KVJ4H8PTJDJCGHHFRB8CD3GP | True-3D Qwen-like 1.2B Cx8 retry finished cleanly and is plotted. |
| `ds-1p2b-cx8-ds2-sh-lr4e-4-r1` | done | finalized 2026-06-29 13:25 | https://beaker.org/ex/01KVV2CVAH6ZYWMRYC8TRDS4DJ | dense2 1.2B Cx8 finished cleanly and is plotted. |
| `ds-1p2b-cx8-ds4-sh-lr4e-4-r1` | done | finalized 2026-06-30 04:20 | https://beaker.org/ex/01KVV2F1R6M48R02BKM9RVJZH0 | dense4 1.2B Cx8 finished cleanly and is plotted. |
| `se-1p2b-cx8-se0m9-lr4e-4-r1` | done | finalized 2026-06-30 09:43 | https://beaker.org/ex/01KVV2FTHMVKP4ARF5B2A86DN5 | Shared-expert 1.2B Cx8 finished cleanly and is plotted. |

## Known Plotting Issues

### Dense Schedule W&B History Fetches

On 2026-06-29, the previously blocked full W&B `scan_history` calls for `5x1zju17`, `abbmdfx0`, `13sr2oht`, and `4x0anaih` succeeded and those runs are now in the canonical dense-schedule plots. The sampled fallback values below are retained only as a record of the temporary diagnostic path; do not mix sampled fallback values into canonical plots unless we explicitly decide to change plotting policy.

Current dense-schedule history resolution:

| Run ID | Cell | Beaker state | Resolution | Note |
| --- | --- | --- | --- | --- |
| `qpp6fidz` | 1.2B Cx8 dense0 | finished | full history cached | Exact avg250M plotted: 2.0809. |
| `rn5yr28o` | 1.2B Cx4 dense2 | finished | full history cached | Exact avg250M plotted: 2.1557. |
| `2i9wpg3j` | 1.2B Cx4 dense4 | finished | tail history cached | Full-history scan still times out, but exact tail scan over final 2,001 steps matches full-cache validation on `rn5yr28o`; exact avg250M plotted: 2.1568. |

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
