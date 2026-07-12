# Ladder Run Tracker

Last updated: 2026-07-12 06:35 UTC.

This table is a scan-friendly status matrix for planned ladder cells. It is separate from `RUNS.md` (chronological launch/status log) and `PLOTTED_RESULTS.md` (finished-only plotted rows and losses).

Main experiment categories: baseline, dense schedule, expert granularity, integration candidates, midtraining, Qwen-like, shared expert, and total sparsity. Rows marked diagnostic are tracked for context but are not part of the main full-grid completion target.

Legend: `done` = at least one finished/plotted run exists; `run` = currently running in Beaker; `queued` = created/scheduled but not started; `todo` = planned/not started; `hold` = intentionally not prioritized yet.

## Current Full-Grid Gaps

| Experiment | Remaining not-yet-queued / not-started cells | Notes |
| --- | --- | --- |
| Total sparsity | 1.2B Cx1/2/4/8 for high total 96E/top4 and huge total 192E/top4 | 275M, 480M, and 810M are done. |
| Integration candidates | None for pretraining | 275M LR grid plus 480M/810M/1.2B wide/deep promotions are finished. |
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
| Integration candidates | wide 256E/top8 + shared + dense1 | Cx1/Cx2/Cx4/Cx8 done/plotted and bracketed | Cx1/Cx2/Cx4/Cx8 done/plotted | Cx1/Cx2/Cx4/Cx8 done/plotted | done Cx1/2/4/8 | 480M and 810M wide promotions beat same-LR baseline; 1.2B wide is finished and ready for MT/evals. |
| Integration candidates | deep 256E/top8 + shared + dense1 | Cx1/2/4/8 done/plotted | Cx1/Cx2/Cx4/Cx8 done/plotted | Cx1/Cx2/Cx4/Cx8 done/plotted | done Cx1/2/4/8 | 480M/810M deep beat wide and baseline at same LR; 1.2B deep Cx8 finished and is ready for resumed MT. |

## Midtraining Tracker

Midtraining uses semantic run names that omit GPU count, node count, microbatch,
and batch size. Systems settings live here and in W&B tags so runs can resume
cleanly if we adjust hardware later.

Default 275M midtraining settings for the first full grid: 100B tokens, sequence
length 8192, global batch seq 128 (1,048,576 tokens), 1 node, 4 GPUs, EP1,
microbatch 8, fresh optimizer state, 2000-step warmup then constant LR.

| Source | Source checkpoint | LR grid | State | Beaker | Notes |
| --- | --- | --- | --- | --- | --- |
| 275M baseline Cx1 | `olmoe3-tiny-275m-cx1-b256k-gpu2-ep1mb16-lr2e-3-r2/step15365` | `2e-4`, `4e-4`, `8e-4`, `1.6e-3` | done | [2e-4](https://beaker.org/ex/01KWWM1043JEC9MC3PV7PXQ745), [4e-4](https://beaker.org/ex/01KWWM1AVQXMJ3JBJQ1W2G8YAV), [8e-4](https://beaker.org/ex/01KWWM1N0EQDMCFER90NVP9QW0), [1.6e-3](https://beaker.org/ex/01KWWM1ZXN5R5XWK00GH0WA36G) | All four training jobs finished. Tests midtraining LR transfer from the low-data optimal baseline checkpoint. |
| 275M baseline Cx2 | `olmoe3-tiny-275m-cx2-b384k-gpu2-ep1mb8-lr1.8e-3-r3/step20486` | `1.8e-4` | done | [1.8e-4](https://beaker.org/ex/01KWZ8T9BZ3B869VZ878FNNQ8T) | Single-point 10% PT-LR run finished 2026-07-08; final-checkpoint eval backfill uploaded 2026-07-10. |
| 275M baseline Cx4 | `olmoe3-tiny-275m-cx4-b512k-gpu4-ep1mb16-lr1.5e-3/step30729` | `1.5e-4` | done | [1.5e-4](https://beaker.org/ex/01KWZ8T9DZN8Y4VD63AP1AN387) | Single-point 10% PT-LR run finished 2026-07-08; final-checkpoint eval backfill uploaded 2026-07-10. |
| 275M baseline Cx8 | `olmoe3-tiny-275m-cx8-b768k-gpu4-ep1mb8-lr1.6e-3-r2/step40971` | `2e-4`, `4e-4`, `8e-4`, `1.6e-3` | done | [2e-4](https://beaker.org/ex/01KWWM10ANMMW2YTNN6RKJBGE7), [4e-4](https://beaker.org/ex/01KWWM1AK89SKDA1KGCX5D8SMM), [8e-4](https://beaker.org/ex/01KWWM1P9TXRSMDV5DH9EF4KXM), [1.6e-3](https://beaker.org/ex/01KWWM213KMG47KADPK5Q67GJP) | All four training jobs finished. Tests midtraining LR transfer from the high-data optimal baseline checkpoint. |
| 275M integration wide Cx1 | `integration/int-275m-cx1-intw256e8k-lr1.6e-3-r1/step15499` | `2e-4` | done | [2e-4 r3](https://beaker.org/ex/01KX1WZ551YCZYTHXRQEDX1WK1) | Canonical replacement for failed r1, launched from pushed commit `34f5d5a` with `integration_midtraining_ladder.py` and fresh optimizer-state loading. Finished 2026-07-09. |
| 275M integration wide Cx2 | `integration/int-275m-cx2-intw256e8k-lr1.6e-3-r1/step20665` | `1.8e-4` | done | [1.8e-4 r3](https://beaker.org/ex/01KX1WZGKE5K5GF4SZ98G7SVJD) | Canonical replacement for failed r1, launched from pushed commit `34f5d5a` with `integration_midtraining_ladder.py` and fresh optimizer-state loading. Finished 2026-07-09. |
| 275M integration wide Cx4 | `integration/int-275m-cx4-intw256e8k-lr8e-4-r1/step30997` | `1.5e-4` | done | [1.5e-4 r3](https://beaker.org/ex/01KX1WZXCBX8VFNFFZK2DT50JB) | Canonical replacement for failed r1, launched from pushed commit `34f5d5a` with `integration_midtraining_ladder.py` and fresh optimizer-state loading. Finished 2026-07-10. |
| 275M integration wide Cx8 | `integration/int-275m-cx8-intw256e8k-lr8e-4-r1/step41329` | `1.6e-4` | done | [1.6e-4 r3](https://beaker.org/ex/01KX1X0AJAMCEE5MSXBP33PHNG) | Canonical replacement for failed r1, launched from pushed commit `34f5d5a` with `integration_midtraining_ladder.py` and fresh optimizer-state loading. Finished 2026-07-10. |
| 275M integration deep Cx1 | `integration/int-275m-cx1-intd256e8k-lr1.6e-3-r1/step15130` | `2e-4` | done | [2e-4 r3](https://beaker.org/ex/01KX1X0QVHZ7X0HSF0YNJWEBBH) | Canonical replacement for failed r1, launched from pushed commit `34f5d5a` with `integration_midtraining_ladder.py` and fresh optimizer-state loading. Finished 2026-07-10. |
| 275M integration deep Cx2 | `integration/int-275m-cx2-intd256e8k-lr1.6e-3-r1/step20173` | `1.8e-4` | done | [1.8e-4 r3](https://beaker.org/ex/01KX1X14K49GJN4KS9QQ3JEQ1E) | Canonical replacement for failed r1, launched from pushed commit `34f5d5a` with `integration_midtraining_ladder.py` and fresh optimizer-state loading. Finished 2026-07-10. |
| 275M integration deep Cx4 | `integration/int-275m-cx4-intd256e8k-lr1.6e-3-r1/step30259` | `1.5e-4` | done | [1.5e-4 r3](https://beaker.org/ex/01KX1X1GPC3N51C7E8AW2YXN63) | Canonical replacement for failed r1, launched from pushed commit `34f5d5a` with `integration_midtraining_ladder.py` and fresh optimizer-state loading. Finished 2026-07-10. |
| 275M integration deep Cx8 | `integration/int-275m-cx8-intd256e8k-lr1.6e-3-r1/step40345` | `1.6e-4` | done | [1.6e-4 r3](https://beaker.org/ex/01KX1X203AABJ29GHKST48J5E9) | Canonical replacement for failed r1, launched from pushed commit `34f5d5a` with `integration_midtraining_ladder.py` and fresh optimizer-state loading. Finished 2026-07-10. |
| 480M integration wide Cx8 | `integration/int-480m-cx8-intw256e8k-lr8e-4-r1/step78042` | `8e-5` | done | [relaunch](https://beaker.org/ex/01KX6K2NTZZRVE98HCQWE2DR64) | Finished 2026-07-12 at `step63579`. Final-checkpoint eval backfill launched as [01KXAG736RGRPJ23ZF172DTHBH](https://beaker.org/ex/01KXAG736RGRPJ23ZF172DTHBH); HF conversion launched as [01KXAG811R26WXR368B2N3A3B0](https://beaker.org/ex/01KXAG811R26WXR368B2N3A3B0). |
| 810M integration wide Cx8 | `integration/int-810m-cx8-intw256e8k-lr4e-4-r1/step141423` | `4e-5` | done | [relaunch](https://beaker.org/ex/01KX6K330J3Z5CSDY518BGJ05B) | Finished 2026-07-12 at `step47684`. Final-checkpoint eval backfill launched as [01KXAEK0VQBM0S74A8KZXXHWWM](https://beaker.org/ex/01KXAEK0VQBM0S74A8KZXXHWWM); HF conversion launched as [01KXAEMCSE3T436X2K4DGK0FYH](https://beaker.org/ex/01KXAEMCSE3T436X2K4DGK0FYH). |
| 1.2B integration wide Cx8 | `integration/int-1p2b-cx8-intw256e8k-lr4e-4-r2/step217870` | `4e-5` | run | [PT-settings relaunch](https://beaker.org/ex/01KX7032K1GXDKQR2D9FAA0CBF) | Previous EP1/MB4 relaunch OOMed during checkpoint restore/model-buffer sync. Retried with the exact successful PT systems settings: GBS seq 96, 1 node x 8 GPUs, EP8 / MB3. |
| 480M integration deep Cx8 | `integration/int-480m-cx8-intd256e8k-lr8e-4-r1/step78659` | `8e-5` | run | [relaunch](https://beaker.org/ex/01KX6K41XG75V1DQ9NWZB988GQ) | Initial promoted Cx8 MT failed during checkpoint save under Weka space pressure; relaunched with same semantic name/save folder to resume from partial checkpoint. |
| 810M integration deep Cx8 | `integration/int-810m-cx8-intd256e8k-lr4e-4-r1/step138619` | `4e-5` | run | [relaunch](https://beaker.org/ex/01KX6K4GKDHQS05727Q9DH0NF4) | Initial promoted Cx8 MT failed during checkpoint save under Weka space pressure; relaunched with same semantic name/save folder to resume from partial checkpoint. |
| 1.2B integration deep Cx8 | `integration/int-1p2b-cx8-intd256e8k-lr4e-4-r2/step218156` | `4e-5` | run | [PT-settings relaunch](https://beaker.org/ex/01KX703FE8T96HJM4EFG90KHVF) | Previous EP1/MB4 relaunch reached compiled forward then OOMed. Retried with the exact successful PT systems settings: GBS seq 96, 1 node x 8 GPUs, EP8 / MB3. Started 2026-07-10. |

Final-checkpoint eval backfills are eval-only jobs over the final midtraining checkpoint. Earlier attempts from commits `430f233c`, `571c0984`, and `ac32eb76` failed before eval due to duplicate evaluator callbacks or dataloader restore mismatches and should be ignored. The fixed backfills build the real midtraining source-mixture dataloader for eval. On 2026-07-07, `copy_eval_backfills_to_wandb.py --only mt-eval` copied the original eight 275M Cx1/Cx8 backfills to their source W&B runs; on 2026-07-10 it copied the six new Cx2/Cx4/480M/810M backfills as well. On 2026-07-12, it copied the 1.2B baseline Cx1/Cx8 and 275M integration wide/deep backfills.

| Source | LR | Eval backfill | State | Notes |
| --- | --- | --- | --- | --- |
| Cx1 | `2e-4` | [01KWYWXRZQJB7RGNAHEE0NCR65](https://beaker.org/ex/01KWYWXRZQJB7RGNAHEE0NCR65) | uploaded | Eval metrics copied to source run. |
| Cx1 | `4e-4` | [01KWYWY36BF88883A6EP91T6YQ](https://beaker.org/ex/01KWYWY36BF88883A6EP91T6YQ) | uploaded | Eval metrics copied to source run. |
| Cx1 | `8e-4` | [01KWYWYE8N65R781R8H59P6EM2](https://beaker.org/ex/01KWYWYE8N65R781R8H59P6EM2) | uploaded | Eval metrics copied to source run. |
| Cx1 | `1.6e-3` | [01KWZ31W7SGHQFY1SKCE0M6APR](https://beaker.org/ex/01KWZ31W7SGHQFY1SKCE0M6APR) | uploaded | Eval metrics copied to source run. |
| Cx8 | `2e-4` | [01KWYWYT764G3SD3A6G7V93MFY](https://beaker.org/ex/01KWYWYT764G3SD3A6G7V93MFY) | uploaded | Eval metrics copied to source run. |
| Cx8 | `4e-4` | [01KWYWZ574B2TG9183DVDPPVQC](https://beaker.org/ex/01KWYWZ574B2TG9183DVDPPVQC) | uploaded | Eval metrics copied to source run. |
| Cx8 | `8e-4` | [01KWYWZGNQ3KK63CDQPB4RFYAT](https://beaker.org/ex/01KWYWZGNQ3KK63CDQPB4RFYAT) | uploaded | Eval metrics copied to source run. |
| Cx8 | `1.6e-3` | [01KWZ3287FMKN570V86JD3XANK](https://beaker.org/ex/01KWZ3287FMKN570V86JD3XANK) | uploaded | Eval metrics copied to source run. |
| 275M Cx2 | `1.8e-4` | [01KX40CX1CHF7CFFQSDZK4Z4RE](https://beaker.org/ex/01KX40CX1CHF7CFFQSDZK4Z4RE) | uploaded | Final-checkpoint eval metrics copied to source W&B run on 2026-07-10. |
| 275M Cx4 | `1.5e-4` | [01KX40D8PFC9Z920Q3T70TGMSQ](https://beaker.org/ex/01KX40D8PFC9Z920Q3T70TGMSQ) | uploaded | Final-checkpoint eval metrics copied to source W&B run on 2026-07-10. |
| 480M Cx1 | `1.2e-4` | [01KX40CX2BCWQZ6N1AAAQ42J0A](https://beaker.org/ex/01KX40CX2BCWQZ6N1AAAQ42J0A) | uploaded | Final-checkpoint eval metrics copied to source W&B run on 2026-07-10. |
| 480M Cx8 | `8e-5` | [01KX40D8N8FWQKGJXQ5716CKHG](https://beaker.org/ex/01KX40D8N8FWQKGJXQ5716CKHG) | uploaded | Final-checkpoint eval metrics copied to source W&B run on 2026-07-10. |
| 810M Cx1 | `6e-5` | [01KX40CX1TDFGF4115E01P6Y3Q](https://beaker.org/ex/01KX40CX1TDFGF4115E01P6Y3Q) | uploaded | Final-checkpoint eval metrics copied to source W&B run on 2026-07-10. |
| 810M Cx8 | `4e-5` | [01KX40D8K5SZJG3G4HNP8VDX67](https://beaker.org/ex/01KX40D8K5SZJG3G4HNP8VDX67) | uploaded | Final-checkpoint eval metrics copied to source W&B run on 2026-07-10. |
| 1.2B Cx8 | `4e-5` | [01KX4WN3XFYSZAGH33KWYF62TV](https://beaker.org/ex/01KX4WN3XFYSZAGH33KWYF62TV) | uploaded | Final-checkpoint eval metrics copied to source W&B run on 2026-07-12. |
| 275M integration wide Cx1 | `2e-4` | [01KX4WK50SP2Z3S81H1597P5Q5](https://beaker.org/ex/01KX4WK50SP2Z3S81H1597P5Q5) | uploaded | Final-checkpoint eval metrics copied to source W&B run on 2026-07-12. |
| 275M integration wide Cx2 | `1.8e-4` | [01KX4WKJCYDT2XQAYJFCS9PZJD](https://beaker.org/ex/01KX4WKJCYDT2XQAYJFCS9PZJD) | uploaded | Final-checkpoint eval metrics copied to source W&B run on 2026-07-12. |
| 275M integration wide Cx4 | `1.5e-4` | [01KX4WKZYZJG8W8DPD3F7M1XXH](https://beaker.org/ex/01KX4WKZYZJG8W8DPD3F7M1XXH) | uploaded | Final-checkpoint eval metrics copied to source W&B run on 2026-07-12. |
| 275M integration wide Cx8 | `1.6e-4` | [01KX4WME9TKAQHGM8CE3073HHZ](https://beaker.org/ex/01KX4WME9TKAQHGM8CE3073HHZ) | uploaded | Final-checkpoint eval metrics copied to source W&B run on 2026-07-12. |
| 275M integration deep Cx1 | `2e-4` | [01KX5CM5NCAGRQ97Z51CSCVXR3](https://beaker.org/ex/01KX5CM5NCAGRQ97Z51CSCVXR3) | uploaded | Final-checkpoint eval metrics copied to source W&B run on 2026-07-12. |
| 275M integration deep Cx2 | `1.8e-4` | [01KX5CMGM8BKX6JYHXWQ3A1CXJ](https://beaker.org/ex/01KX5CMGM8BKX6JYHXWQ3A1CXJ) | uploaded | Final-checkpoint eval metrics copied to source W&B run on 2026-07-12. |
| 275M integration deep Cx4 | `1.5e-4` | [01KX5CMWZR8PJSJEG56DCB9455](https://beaker.org/ex/01KX5CMWZR8PJSJEG56DCB9455) | uploaded | Final-checkpoint eval metrics copied to source W&B run on 2026-07-12. |
| 275M integration deep Cx8 | `1.6e-4` | [01KX5CN86SGER9E88YASSS7J06](https://beaker.org/ex/01KX5CN86SGER9E88YASSS7J06) | uploaded | Final-checkpoint eval metrics copied to source W&B run on 2026-07-12. |
| 1.2B baseline Cx1 | `4e-5` | [01KX5CNTTRBMEXBA1XJ1JNM0ZR](https://beaker.org/ex/01KX5CNTTRBMEXBA1XJ1JNM0ZR) | uploaded | Final-checkpoint eval metrics copied to source W&B run on 2026-07-12. |
| 480M integration wide Cx8 | `8e-5` | [01KXAG736RGRPJ23ZF172DTHBH](https://beaker.org/ex/01KXAG736RGRPJ23ZF172DTHBH) | launched | Final-checkpoint eval backfill launched 2026-07-12 from `step63579`; copy to source W&B after it finishes. |
| 810M integration wide Cx8 | `4e-5` | [01KXAEK0VQBM0S74A8KZXXHWWM](https://beaker.org/ex/01KXAEK0VQBM0S74A8KZXXHWWM) | launched | Final-checkpoint eval backfill launched 2026-07-12 from `step47684`; copy to source W&B after it finishes. |

Tentative larger-model midtraining batch targets: 480M uses global batch seq 192,
810M uses 256, and 1.2B uses 384. These require smoke tests before promotion.
The working LR rule is 10% of the canonical baseline best observed pretraining
LR at the matching `(model size, Cx)` point.

| Model size | Smoke source | Smoke LR | Max tokens | Global batch seq | GPUs | EP / MB | State | Beaker | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| 480M | baseline Cx8 best checkpoint | `8e-5` | `2B` | `192` | 4 | EP1 / MB8 | passed, stopped | [01KWZ9B2HS4RVK6ZFC5V80YGNT](https://beaker.org/ex/01KWZ9B2HS4RVK6ZFC5V80YGNT) | Started stepping with compile-on; stopped after smoke success before full sweep. |
| 810M | baseline Cx8 best checkpoint | `4e-5` | `2B` | `256` | 8 | EP1 / MB4 | passed, stopped | [01KWZ9B9KHRJTW139P8PMV6A4F](https://beaker.org/ex/01KWZ9B9KHRJTW139P8PMV6A4F) | Started stepping with compile-on; stopped after smoke success before full sweep. |
| 1.2B | baseline Cx8 best checkpoint | `4e-5` | `2B` | `384` | 8 | EP1 / MB4 | passed, stopped | [01KWZ9B4C3P04P6R5K63W8TMDF](https://beaker.org/ex/01KWZ9B4C3P04P6R5K63W8TMDF) | Started stepping with compile-on; stopped after smoke success before full sweep. |

Full larger-model midtraining sweep launched on 2026-07-07. On 2026-07-08, Cx2/Cx4 for 480M/810M/1.2B were stopped to wait for the 275M midtraining signal before spending larger-scale compute. As of 2026-07-10, 480M and 810M Cx1/Cx8 training is done and included in the midtraining dashboard with final-checkpoint eval metrics uploaded; 1.2B Cx8 training is done but still needs a final-checkpoint eval backfill, and 1.2B Cx1 remains running. All runs use 100B
midtraining tokens, sequence length 8192, Titan urgent, compile-on, fresh
optimizer state, weight-only load from the canonical baseline checkpoint, 2000
step warmup then constant LR, and the 10% baseline pretraining LR rule.

| Source | Source checkpoint | LR | Global batch seq | GPUs | EP / MB | State | Beaker |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| 480M baseline Cx1 | `m480-cx1-b256k-gpu4-ep1mb8-lr1.2e-3-r1/step29022` | `1.2e-4` | `192` | 4 | EP1 / MB8 | done | [01KWZARWH7XAS4MD2238VRKP0Y](https://beaker.org/ex/01KWZARWH7XAS4MD2238VRKP0Y) |
| 480M baseline Cx2 | `m480-cx2-b384k-gpu4-ep1mb4-lr9e-4-r1/step38696` | `9e-5` | `192` | 4 | EP1 / MB8 | stopped | [01KWZARZ2XT6HP3HZ4AK0F5EKT](https://beaker.org/ex/01KWZARZ2XT6HP3HZ4AK0F5EKT) |
| 480M baseline Cx4 | `m480-cx4-b512k-gpu4-ep1mb8-lr8e-4-r1/step58044` | `8e-5` | `192` | 4 | EP1 / MB8 | stopped | [01KWZARWY3JYNCT8HAMYAD17J7](https://beaker.org/ex/01KWZARWY3JYNCT8HAMYAD17J7) |
| 480M baseline Cx8 | `m480-cx8-b768k-gpu8-ep1mb4-lr8e-4-r1/step77392` | `8e-5` | `192` | 4 | EP1 / MB8 | done | [01KWZARZ2T7FZDH42Q2VS92WXN](https://beaker.org/ex/01KWZARZ2T7FZDH42Q2VS92WXN) |
| 810M baseline Cx1 | `olmoe3-moe-a0-810m-cx1-b256k-gpu4-ep1mb4-lr6e-4-r1/step52648` | `6e-5` | `256` | 8 | EP1 / MB4 | done | [01KWZAT4PF87PYA96JT7ZXSQKX](https://beaker.org/ex/01KWZAT4PF87PYA96JT7ZXSQKX) |
| 810M baseline Cx2 | `olmoe3-moe-a0-810m-cx2-b384k-gpu8-ep1mb2-lr5.6e-4-r3/step70197` | `5.6e-5` | `256` | 8 | EP1 / MB4 | stopped | [01KWZAT5N7AJ6C1HTTRMDD8WMT](https://beaker.org/ex/01KWZAT5N7AJ6C1HTTRMDD8WMT) |
| 810M baseline Cx4 | `olmoe3-moe-a0-810m-cx4-b512k-gpu8-ep1mb4-lr4e-4-r1/step105295` | `4e-5` | `256` | 8 | EP1 / MB4 | stopped | [01KWZAT4PE90E0X0N6Z126Q8SR](https://beaker.org/ex/01KWZAT4PE90E0X0N6Z126Q8SR) |
| 810M baseline Cx8 | `olmoe3-moe-a0-810m-cx8-b768k-gpu8-ep1mb4-lr4e-4-r1/step140394` | `4e-5` | `256` | 8 | EP1 / MB4 | done | [01KWZAT4PQ0NWD20VS0XT12ZF5](https://beaker.org/ex/01KWZAT4PQ0NWD20VS0XT12ZF5) |
| 1.2B baseline Cx1 | `olmoe3-moe-a0-1p2b-cx1-b256k-gpu8-ep1mb2-lr4e-4-r1/step81190` | `4e-5` | `384` | 8 | EP1 / MB4 | done | [01KWZAVZFS1FMR59ASRVH7VD4X](https://beaker.org/ex/01KWZAVZFS1FMR59ASRVH7VD4X) |
| 1.2B baseline Cx2 | `olmoe3-moe-a0-1p2b-cx2-b384k-lr6e-4-r1/step108253` | `6e-5` | `384` | 8 | EP1 / MB4 | stopped | [01KWZAV9B2740S4NZDAZJW5A0R](https://beaker.org/ex/01KWZAV9B2740S4NZDAZJW5A0R) |
| 1.2B baseline Cx4 | `olmoe3-moe-a0-1p2b-cx4-b512k-gpu8-ep1mb2-lr3e-4-r1/step162379` | `3e-5` | `384` | 8 | EP1 / MB4 | stopped | [01KWZAVC810DHNY5VE958S6DTG](https://beaker.org/ex/01KWZAVC810DHNY5VE958S6DTG) |
| 1.2B baseline Cx8 | `olmoe3-moe-a0-1p2b-cx8-b768k-gpu32-ep1mb1-lr4e-4-r1/step216505` | `4e-5` | `384` | 8 | EP1 / MB4 | done | [01KWZAV9AGDSD3PTS5RAM7G5N2](https://beaker.org/ex/01KWZAV9AGDSD3PTS5RAM7G5N2) |

## Midtraining HF Conversion / OLMoBase Evals

Initial OLMoBase eval target set used the selected 275M baseline midtraining LR, `2e-4`, for both Cx1 and Cx8. HF checkpoints are written under `hf-checkpoints/midtraining/`. On 2026-07-10, the Cx8 MT set was extended to finished baseline sizes plus the finished 275M integration-wide MT checkpoint.

| Source | HF conversion | OLMoBase eval | State | Notes |
| --- | --- | --- | --- | --- |
| `mt-275m-baseline-cx1-lr2e-4-r1/step95368` | [01KX02AE8YRXS7Q8SGV0TAMEN7](https://beaker.org/ex/01KX02AE8YRXS7Q8SGV0TAMEN7) | [01KX02MYSA36TFKBT99XAE0XZ6](https://beaker.org/ex/01KX02MYSA36TFKBT99XAE0XZ6) | done | Converted HF path: `hf-checkpoints/midtraining/mt-275m-baseline-cx1-lr2e-4-r1/step95368`. |
| `mt-275m-baseline-cx8-lr2e-4-r1/step95368` | [01KX02AFMHGPSH4ZQMS0A5BEF7](https://beaker.org/ex/01KX02AFMHGPSH4ZQMS0A5BEF7) | [01KX02N80T6NXX8TZEJM8TC0Z5](https://beaker.org/ex/01KX02N80T6NXX8TZEJM8TC0Z5) | done | Converted HF path: `hf-checkpoints/midtraining/mt-275m-baseline-cx8-lr2e-4-r1/step95368`. |
| `mt-480m-baseline-cx8-lr8e-5-r1/step63579` | [01KX4WQ46EHZTA76REXB4DWX8E](https://beaker.org/ex/01KX4WQ46EHZTA76REXB4DWX8E) | [01KX4X15K8VHV25ES8YSPF0AET](https://beaker.org/ex/01KX4X15K8VHV25ES8YSPF0AET) | done | OLMoBase finalized 2026-07-10. |
| `mt-810m-baseline-cx8-lr4e-5-r1/step47684` | [01KX4WQ5D3C5E2QED9S4NKZXWG](https://beaker.org/ex/01KX4WQ5D3C5E2QED9S4NKZXWG) | [01KX4X1GRMAR78VYH0XZVJX7MX](https://beaker.org/ex/01KX4X1GRMAR78VYH0XZVJX7MX) | done | OLMoBase finalized 2026-07-10. |
| `mt-1p2b-baseline-cx8-lr4e-5-r1/step31790` | [01KX4WQ6MBG2Z525R1NRN6FHEH](https://beaker.org/ex/01KX4WQ6MBG2Z525R1NRN6FHEH) | [01KX4X6FYR1DTE1VMY3FEGT5XC](https://beaker.org/ex/01KX4X6FYR1DTE1VMY3FEGT5XC) | done | OLMoBase finalized 2026-07-10. |
| `mt-275m-intw256e8k-cx8-lr1p6e-4-r3/step95368` | [01KX4WQ7T1G1SC89E0PEHW1RRS](https://beaker.org/ex/01KX4WQ7T1G1SC89E0PEHW1RRS) | [01KX4XBGFP5A0XE1JFQT2NWCVM](https://beaker.org/ex/01KX4XBGFP5A0XE1JFQT2NWCVM) | done | OLMoBase finalized 2026-07-10. |
| `mt-275m-intd256e8k-cx8-lr1p6e-4-r3/step95368` | [01KX5DTZQJKG11EGTERPKWXA0F](https://beaker.org/ex/01KX5DTZQJKG11EGTERPKWXA0F) | [01KX5FF8P4F3GS5RRH5M29AFCW](https://beaker.org/ex/01KX5FF8P4F3GS5RRH5M29AFCW) | done | OLMoBase finalized 2026-07-10. |
| `mt-480m-intw256e8k-cx8-lr8e-5-r1/step63579` | [01KXAG811R26WXR368B2N3A3B0](https://beaker.org/ex/01KXAG811R26WXR368B2N3A3B0) | [01KXAGZ02JQ3M71M6MBE9PBBZT](https://beaker.org/ex/01KXAGZ02JQ3M71M6MBE9PBBZT) | running | HF conversion wrote the target HF checkpoint; OLMoBase launched 2026-07-12 in `ai2/olmo-instruct`. |
| `mt-810m-intw256e8k-cx8-lr4e-5-r1/step47684` | [01KXAEMCSE3T436X2K4DGK0FYH](https://beaker.org/ex/01KXAEMCSE3T436X2K4DGK0FYH) | [01KXAFXV1EAX2FC89C3YGZEJ61](https://beaker.org/ex/01KXAFXV1EAX2FC89C3YGZEJ61) | running | HF conversion finished 2026-07-12; OLMoBase launched in the MoE workspace. |

## Active / Queued Beaker Surface

Bounded status pass on 2026-07-09 18:02 UTC checked only runs that were previously running, queued, created, or finished-unplotted in this table / `RUNS.md`; it did not scan the full historical W&B/Beaker surface.

| Run(s) | State | Latest timestamp UTC | Beaker | Notes |
| --- | --- | --- | --- | --- |
| `int-810m-cx{1,2,4,8}-intw256e8k-baseline-LR-r1` | done | plotted 2026-07-09 | https://beaker.org/ex/01KWGQP0NGMXDEN2PBRGAXJZ7R | 810M wide integration promoted single points on Titan urgent, compile-on. All four Cx runs succeeded and are plotted after the latest tail-cache refresh. GBS seq 32/48/64/96; GPUs 8/8/8/8; MB 4/2/4/4. |
| `int-810m-cx{1,2,4,8}-intd256e8k-baseline-LR-r1` | done | plotted 2026-07-09 | https://beaker.org/ex/01KWGQQVD0TE5ZY5K5T05GEYAK | 810M deep integration promoted single points on Titan urgent, compile-on. All four Cx runs succeeded and are plotted after the latest tail-cache refresh. GBS seq 32/48/64/96; GPUs 8/8/8/8; MB 4/2/4/4. Current Beaker IDs: Cx1 `01KWGQQVD0TE5ZY5K5T05GEYAK`; Cx2 `01KWHQZQZ811104C16PJ4GTWG7`; Cx4 `01KWHR02WQ7S4FCQ77WJNFSA5S`; Cx8 `01KWHR0DANF34FXQ236WKHHGMM`. |
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
| `int-275m-cx4-intw256e16k-lr8e-4-r1` | submitted | launched 2026-07-12 | https://beaker.org/ex/01KXA424V6TFS1GQ8A85PJEFRT | High-active wide integration diagnostic: 256E/top16, 0.5d routed/shared experts, dense1. Uses Cx4 LR `8e-4`, 1 node x 8 GPUs, GBS seq 64, MB4, and explicit `--max-duration-tokens=32502185984` to match the 275M wide Cx8 token budget within one half Cx4 batch. |
| `int-275m-cx8-intw256e8k-lr4e-4-r1` | done | plotted 2026-07-03 00:20 | https://beaker.org/ex/01KWFRYPK82AC42HWXR0HNRE3G | Cold-side wide 275M Cx8 follow-up finished. `4e-4` avg250M `2.4359` vs `8e-4` avg250M `2.4193`, so wide Cx8 remains best observed at `8e-4`. |
| `int-480m-cx{1,2}-intd256e8k-baseline-LR-r1` | done | plotted 2026-07-02 02:00 | https://beaker.org/ex/01KWF7Z3GFWM8JS31NB0P9M516 | 480M deep Cx1/Cx2 finished. Cx1 `1.2e-3` avg250M `2.5291`; Cx2 `9e-4` avg250M `2.4091`. Both beat wide and same-LR baseline. |
| `int-480m-cx{4,8}-intd256e8k-baseline-LR-r1` | done | plotted 2026-07-03 00:20 | https://beaker.org/ex/01KWF804APPWNTTXT0B118G5MB | 480M deep Cx4/Cx8 finished. Cx4 `8e-4` avg250M `2.3207`; Cx8 `8e-4` avg250M `2.2380`. Both beat wide and same-LR baseline. |
| `int-480m-cx{1,2}-intw256e8k-baseline-LR-r1` | done | plotted 2026-07-01 10:17 | https://beaker.org/ex/01KWE2XDE9NATMWCWKAH9X29JT | 480M wide Cx1/Cx2 finished. Cx1 `1.2e-3` avg250M `2.5433` vs baseline same-LR `2.5636`; Cx2 `9e-4` avg250M `2.4239` vs baseline same-LR `2.4630`. |
| `int-480m-cx4-intw256e8k-baseline-LR-r1` | done | plotted 2026-07-02 02:00 | https://beaker.org/ex/01KWE2Y61CJEQEDW57MDJEPRDH | 480M wide Cx4 `8e-4` avg250M `2.3300` vs baseline same-LR `2.3788`. |
| `int-480m-cx8-intw256e8k-baseline-LR-r1` | done | plotted 2026-07-03 00:20 | https://beaker.org/ex/01KWE2YHFRQHAF56KCK95FPK62 | 480M wide Cx8 `8e-4` avg250M `2.2513` vs baseline same-LR `2.3076`; deep Cx8 is stronger at `2.2380`. |
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
