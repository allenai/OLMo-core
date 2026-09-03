# FLOP-scaling study: FFN routing and KV soft tokens vs dense (Qwen3.5-4B)

Generated 2026-09-02 18:09 from `results/flop_scaling/results35.csv`. Plan: `records/flop-scaling-ffn-kv-plan.md`. Ledger: `debug/flop_scaling/LAUNCH_LEDGER.tsv`.

Run status at generation: 92/93 training runs done (0 failed), 90/92 evals done. Cells missing below are still queued/running on Beaker.

## Summary (written 2026-09-02 18:10; grid 92/93 runs and 87/89 evals done, three outlier-320M tail items pending)

**Question.** At matched *training* FLOPs, does either method beat dense SFT on Qwen3.5-4B short-heavy 2k–32k mixes?

**Answer per task** (f1 = mean over rungs; PF = training PFLOPs at real example lengths; dense anchors at 4–10M tokens added today so every comparison is inside the measured range):

| task | best method arm | matched-compute verdict | FLOPs to the task target, method ÷ dense (Hill fit) |
|---|---|---|---|
| oolong | KV soft tokens, keep 1/6 (gold-blind) | **wins below ~500 PF**: 0.637 @134 PF vs dense 0.624 @243 PF; 0.652 @272 vs dense 0.661 @486; 0.665 @564 vs dense 0.702 @972 | 1.4× at f1 0.68 (fit is flat; direct comparison says ≈0.55–1.2×) |
| contradiction | KV soft tokens, keep gold + 1/3 | **parity to slight loss**: 0.762 @338 PF vs dense 0.772 @337; 0.861 @695 vs dense 0.924 @674; gold-blind collapses (gold pair must be real) | 1.21× at f1 0.85 (FFN stage 1: 1.28×, two-sided: 1.34×) |
| nq | KV soft tokens, gold-blind 1/3 | **dense wins by ~0.03 everywhere**: 0.832 @163 PF vs dense-8M 0.860 @190; 0.878 @509 vs dense-16M 0.878 @379 | 2.5× at f1 0.88 (FFN stage 1: 2.35×) |
| outlier | none | every KV arm dead (≤0.30 at 8k vs dense 0.51–0.89); FFN routing 0.28 vs 0.43 at 64M | — |

**Routed FFN** never reaches compute-optimality here: FFN is 57% of training FLOPs at these lengths, so the saving caps at ~0.70×, and every arm gives up more than that (losses concentrate on the 32k rung). Routing all layers collapses on 30–150-step runs. The two-sided budget lands exactly on target and is the recipe to keep; the L12+ ladder at 0.01 (stage 1) is the better of the two at ≥48M budgets. Deployed inference cost is the one thing FFN routing buys that KV does not.

**KV soft tokens** are task-shaped: they win where the answer is an aggregate over many documents (oolong), tie where two specific documents must survive but usually do (contradiction with gold forced real), trail slightly where one passage must be read verbatim (nq), and fail where every document must be compared (outlier). Forcing gold documents real leaks the answer on id-answer tasks (nq 1/6: 0.603 forced vs 0.728 blind) but is required on contradiction (blind 1/6: 0.053).

**What changed today** (all in §10 of the plan record): a random-router init on every meta-built FFN run (fixed, all FFN numbers re-run), Qwen3 token ids hardcoded in the marker-aware evaluator, a dead rung-file override, an unset-variable death in the eval script, `beaker experiment cancel` being a no-op, marker-data builds that reported success without input, Hub rate limits killing eval starts (tokenizers now staged on weka). Every number above is post-fix.

**Not done / next.** Breadth-matched KV (absolute 128–512 documents, the Qwen3-4B v25 recipe) on 3.5 would test whether contradiction and nq reach parity the way the Qwen3-4B runs did; the all-layer FFN ceiling should be re-measured on the fixed init (v12 rerun, layer-0 exempt from the budget); importance-permuting FFN units before nesting is free and untried.


## Setup

- Model/data/optimizer identical to the prior dense campaigns (`debug/taskscale_lengthmix`, `debug/outlier_lengthmix_scaling`): Qwen3.5-4B (`q35-4b-base-markerfix`), short-heavy 2k–32k length mixes as nested-prefix arms, seq 65536 packed (KV: padded single-example rows, same tokens/step), lr 5e-6, 8 rows/step, 1 epoch. Dense points are those campaigns' numbers, not retrained.
- **FFN routing**: nested-width FFN router over the base FFN (Qwen3.5-4B widths 9728/608/152/38/9/1 + null; the slices share weights). Stage 1 routes layers 12+ (20 of 32) with a one-sided budget hinge at target 0.01; stage 2 warm-starts from stage 1 and routes all layers at 0.02 with the hinge on from step 0; the two-sided arms penalize |cost - target| at 0.10 on layers 12+ (t10) or all layers (a10), no exploration. Scored with routing on (the cut carries to inference).
- **KV soft tokens**: a fixed fraction (1/6 or 1/3) of documents keep real tokens, every other document collapses to one projected soft token in the KV; detached soft KV, no distillation, torch attention backend; scored with plain full attention (training-only saving). kv17/kv33 force the gold documents into the kept set (leaks the answer on id-answer tasks); kvb17/kvb33 are gold-blind (random kept set). Oolong has no gold subset, so its kv17/kv33 are gold-blind by construction.
- **FLOPs**: training FLOPs priced per example at its real length (attention quadratic in the example, not the packed window) from the harvested example lengths; FFN arms scale the FFN share by the mean routed cost the trainer measured; KV arms are metered on their compacted rows. Stage 2 is charged stage 1 + stage 2.
- Accuracy = mean f1 over the task's eval rungs that have a value (same fixed eval sets as the dense campaign; eval_size 500-600 per rung). Where a rung is missing for one point (e.g. contradiction dense 28M lacks 32k) its mean covers fewer rungs -- compare per-rung rows for those.


## contradiction

| arm | 7M tokens<br>mean f1 (PF) | 14M tokens<br>mean f1 (PF) | 28M tokens<br>mean f1 (PF) | 56M tokens<br>mean f1 (PF) |
|---|---|---|---|---|
| dense (prior campaign + 4-10M anchors) | 0.565 (169) | 0.772 (337) | 0.924 (674) | 0.944 (1347) |
| FFN routing, stage 1 (L12+) | – | 0.427 (242) | 0.762 (474) | 0.910 (939) |
| FFN routing, stage 2 (all layers) | – | 0.001 (393) | 0.000 (772) | 0.001 (1533) |
| FFN routing, L12+, two-sided target 0.10 | – | 0.579 (277) | 0.808 (515) | 0.880 (998) |
| FFN routing, all layers, two-sided target 0.10 | – | 0.050 (261) | – | – |
| KV soft-token, keep gold + 1/6 | – | 0.343 (103) | 0.554 (209) | 0.764 (429) |
| KV soft-token, keep gold + 1/3 | – | 0.525 (166) | 0.762 (338) | 0.861 (695) |
| KV soft-token, gold-blind keep 1/6 | – | 0.053 (86) | – | – |
| KV soft-token, gold-blind keep 1/3 | – | 0.268 (151) | 0.577 (312) | 0.761 (638) |

Per-rung f1 (2k, 8k, 16k, 32k):

- dense @ 7M: 0.80 / 0.68 / 0.50 / 0.27
- dense @ 14M: 0.91 / 0.84 / 0.75 / 0.59
- dense @ 28M: 0.97 / 0.93 / 0.87 / –
- dense @ 56M: 0.98 / 0.97 / 0.94 / 0.88
- ffnmoe-s1 @ 14M: 0.70 / 0.54 / 0.30 / 0.17
- ffnmoe-s1 @ 28M: 0.89 / 0.87 / 0.73 / 0.56
- ffnmoe-s1 @ 56M: 0.95 / 0.95 / 0.91 / 0.82
- ffnmoe-s2 @ 14M: 0.00 / 0.00 / 0.00 / 0.00
- ffnmoe-s2 @ 28M: 0.00 / 0.00 / 0.00 / 0.00
- ffnmoe-s2 @ 56M: 0.00 / 0.00 / 0.00 / 0.00
- ffnmoe-t10 @ 14M: 0.80 / 0.70 / 0.51 / 0.30
- ffnmoe-t10 @ 28M: 0.94 / 0.88 / 0.79 / 0.63
- ffnmoe-t10 @ 56M: 0.96 / 0.94 / 0.88 / 0.74
- ffnmoe-a10 @ 14M: 0.14 / 0.05 / 0.01 / 0.00
- kv17 @ 14M: 0.58 / 0.45 / 0.25 / 0.10
- kv17 @ 28M: 0.82 / 0.69 / 0.44 / 0.26
- kv17 @ 56M: 0.93 / 0.87 / 0.73 / 0.53
- kv33 @ 14M: 0.78 / 0.66 / 0.44 / 0.22
- kv33 @ 28M: 0.92 / 0.86 / 0.74 / 0.53
- kv33 @ 56M: 0.96 / 0.92 / 0.84 / 0.72
- kvb17 @ 14M: 0.14 / 0.05 / 0.02 / 0.01
- kvb33 @ 14M: 0.51 / 0.37 / 0.13 / 0.06
- kvb33 @ 28M: 0.79 / 0.71 / 0.50 / 0.32
- kvb33 @ 56M: 0.93 / 0.86 / 0.75 / 0.51

## nq

| arm | 4M tokens<br>mean f1 (PF) | 8M tokens<br>mean f1 (PF) | 16M tokens<br>mean f1 (PF) | 32M tokens<br>mean f1 (PF) | 48M tokens<br>mean f1 (PF) |
|---|---|---|---|---|---|
| dense (prior campaign + 4-10M anchors) | 0.805 (95) | 0.860 (190) | 0.878 (379) | 0.907 (758) | 0.922 (1139) |
| FFN routing, stage 1 (L12+) | – | – | 0.723 (268) | 0.845 (528) | 0.879 (780) |
| FFN routing, stage 2 (all layers) | – | – | 0.046 (435) | 0.526 (859) | 0.607 (1278) |
| FFN routing, L12+, two-sided target 0.10 | – | – | 0.703 (311) | 0.853 (572) | 0.867 (848) |
| FFN routing, all layers, two-sided target 0.10 | – | – | 0.444 (270) | – | – |
| KV soft-token, keep gold + 1/6 | – | – | 0.603 (100) | 0.794 (205) | 0.750 (306) |
| KV soft-token, keep gold + 1/3 | – | – | 0.722 (172) | 0.843 (352) | 0.827 (522) |
| KV soft-token, gold-blind keep 1/6 | – | – | 0.728 (88) | – | – |
| KV soft-token, gold-blind keep 1/3 | – | – | 0.832 (163) | 0.835 (340) | 0.878 (509) |

Per-rung f1 (2k, 8k, 16k, 32k):

- dense @ 4M: 0.95 / 0.82 / 0.77 / 0.68
- dense @ 8M: 0.98 / 0.88 / 0.84 / 0.75
- dense @ 16M: 0.98 / 0.92 / 0.86 / 0.76
- dense @ 32M: 0.98 / 0.93 / 0.89 / 0.84
- dense @ 48M: 0.97 / 0.93 / 0.91 / 0.87
- ffnmoe-s1 @ 16M: 0.97 / 0.85 / 0.64 / 0.43
- ffnmoe-s1 @ 32M: 0.97 / 0.90 / 0.81 / 0.69
- ffnmoe-s1 @ 48M: 0.97 / 0.92 / 0.86 / 0.77
- ffnmoe-s2 @ 16M: 0.13 / 0.03 / 0.01 / 0.01
- ffnmoe-s2 @ 32M: 0.83 / 0.58 / 0.46 / 0.23
- ffnmoe-s2 @ 48M: 0.87 / 0.65 / 0.54 / 0.38
- ffnmoe-t10 @ 16M: 0.97 / 0.86 / 0.66 / 0.32
- ffnmoe-t10 @ 32M: 0.98 / 0.92 / 0.82 / 0.70
- ffnmoe-t10 @ 48M: 0.98 / 0.91 / 0.85 / 0.73
- ffnmoe-a10 @ 16M: 0.81 / 0.49 / 0.31 / 0.17
- kv17 @ 16M: 0.94 / 0.72 / 0.51 / 0.23
- kv17 @ 32M: 0.91 / 0.85 / 0.77 / 0.65
- kv17 @ 48M: 0.93 / 0.85 / 0.70 / 0.52
- kv33 @ 16M: 0.96 / 0.84 / 0.67 / 0.42
- kv33 @ 32M: 0.95 / 0.90 / 0.82 / 0.70
- kv33 @ 48M: 0.94 / 0.89 / 0.82 / 0.67
- kvb17 @ 16M: 0.96 / 0.82 / 0.71 / 0.42
- kvb33 @ 16M: 0.97 / 0.89 / 0.82 / 0.65
- kvb33 @ 32M: 0.95 / 0.90 / 0.82 / 0.67
- kvb33 @ 48M: 0.96 / 0.91 / 0.87 / 0.77

## oolong

| arm | 10M tokens<br>mean f1 (PF) | 20M tokens<br>mean f1 (PF) | 40M tokens<br>mean f1 (PF) | 80M tokens<br>mean f1 (PF) |
|---|---|---|---|---|
| dense (prior campaign + 4-10M anchors) | 0.624 (243) | 0.661 (486) | 0.702 (972) | 0.723 (1946) |
| FFN routing, stage 1 (L12+) | – | 0.585 (336) | 0.655 (666) | 0.679 (1317) |
| FFN routing, stage 2 (all layers) | – | 0.515 (552) | 0.601 (1095) | 0.648 (2175) |
| FFN routing, L12+, two-sided target 0.10 | – | 0.596 (382) | 0.642 (741) | 0.688 (1437) |
| FFN routing, all layers, two-sided target 0.10 | – | 0.508 (336) | – | – |
| KV soft-token, keep gold + 1/6 | – | 0.637 (134) | 0.652 (272) | 0.665 (564) |
| KV soft-token, keep gold + 1/3 | – | 0.641 (227) | 0.663 (457) | 0.674 (?) |

Per-rung f1 (2k, 8k, 16k, 32k):

- dense @ 10M: 0.84 / 0.57 / 0.55 / 0.54
- dense @ 20M: 0.86 / 0.64 / 0.58 / 0.56
- dense @ 40M: 0.89 / 0.67 / 0.64 / 0.61
- dense @ 80M: 0.91 / 0.69 / 0.67 / 0.62
- ffnmoe-s1 @ 20M: 0.82 / 0.54 / 0.50 / 0.47
- ffnmoe-s1 @ 40M: 0.85 / 0.66 / 0.56 / 0.55
- ffnmoe-s1 @ 80M: 0.90 / 0.65 / 0.60 / 0.58
- ffnmoe-s2 @ 20M: 0.75 / 0.45 / 0.45 / 0.41
- ffnmoe-s2 @ 40M: 0.83 / 0.57 / 0.52 / 0.49
- ffnmoe-s2 @ 80M: 0.86 / 0.62 / 0.57 / 0.55
- ffnmoe-t10 @ 20M: 0.84 / 0.57 / 0.50 / 0.48
- ffnmoe-t10 @ 40M: 0.85 / 0.63 / 0.54 / 0.54
- ffnmoe-t10 @ 80M: 0.90 / 0.67 / 0.60 / 0.58
- ffnmoe-a10 @ 20M: 0.76 / 0.43 / 0.42 / 0.42
- kv17 @ 20M: 0.86 / 0.58 / 0.57 / 0.53
- kv17 @ 40M: 0.86 / 0.62 / 0.57 / 0.55
- kv17 @ 80M: 0.86 / 0.63 / 0.59 / 0.58
- kv33 @ 20M: 0.87 / 0.59 / 0.57 / 0.53
- kv33 @ 40M: 0.87 / 0.64 / 0.58 / 0.56
- kv33 @ 80M: 0.89 / 0.64 / 0.60 / 0.57

## outlier

| arm | 8M tokens<br>mean f1 (PF) | 16M tokens<br>mean f1 (PF) | 32M tokens<br>mean f1 (PF) | 64M tokens<br>mean f1 (PF) | 160M tokens<br>mean f1 (PF) | 320M tokens<br>mean f1 (PF) |
|---|---|---|---|---|---|---|
| dense (prior campaign + 4-10M anchors) | 0.194 (190) | 0.263 (380) | 0.338 (760) | 0.428 (1522) | 0.604 (3805) | 0.552 (7608) |
| FFN routing, stage 1 (L12+) | – | 0.056 (272) | 0.134 (532) | 0.280 (1062) | 0.415 (2611) | 0.493 (5261) |
| FFN routing, stage 2 (all layers) | – | 0.019 (442) | 0.089 (866) | 0.165 (1725) | 0.322 (4271) | – |
| FFN routing, L12+, two-sided target 0.10 | – | 0.078 (312) | 0.221 (587) | 0.321 (1124) | 0.378 (2787) | – |
| FFN routing, all layers, two-sided target 0.10 | – | 0.031 (271) | – | – | – | – |
| KV soft-token, keep gold + 1/6 | – | 0.065 (127) | 0.052 (261) | 0.089 (508) | 0.054 (1312) | 0.045 (2641) |
| KV soft-token, keep gold + 1/3 | – | 0.137 (192) | 0.146 (396) | 0.226 (763) | 0.202 (1992) | 0.284 (4013) |
| KV soft-token, gold-blind keep 1/6 | – | 0.033 (96) | – | – | – | – |
| KV soft-token, gold-blind keep 1/3 | – | 0.133 (168) | 0.170 (350) | 0.247 (661) | 0.369 (1753) | – |

Per-rung f1 (2k, 8k, 16k, 32k):

- dense @ 8M: – / 0.40 / 0.17 / 0.01
- dense @ 16M: – / 0.51 / 0.25 / 0.04
- dense @ 32M: – / 0.60 / 0.32 / 0.09
- dense @ 64M: – / 0.72 / 0.45 / 0.11
- dense @ 160M: – / 0.89 / 0.66 / 0.27
- dense @ 320M: – / – / 0.75 / 0.35
- ffnmoe-s1 @ 16M: – / 0.15 / 0.02 / 0.00
- ffnmoe-s1 @ 32M: – / 0.35 / 0.06 / 0.00
- ffnmoe-s1 @ 64M: – / 0.55 / 0.27 / 0.02
- ffnmoe-s1 @ 160M: – / 0.73 / 0.45 / 0.07
- ffnmoe-s1 @ 320M: – / 0.82 / 0.55 / 0.12
- ffnmoe-s2 @ 16M: – / 0.04 / 0.01 / 0.01
- ffnmoe-s2 @ 32M: – / 0.22 / 0.04 / 0.00
- ffnmoe-s2 @ 64M: – / 0.36 / 0.13 / 0.00
- ffnmoe-s2 @ 160M: – / 0.58 / 0.34 / 0.05
- ffnmoe-t10 @ 16M: – / 0.22 / 0.01 / 0.00
- ffnmoe-t10 @ 32M: – / 0.43 / 0.22 / 0.01
- ffnmoe-t10 @ 64M: – / 0.58 / 0.32 / 0.07
- ffnmoe-t10 @ 160M: – / 0.70 / 0.40 / 0.03
- ffnmoe-a10 @ 16M: – / 0.08 / 0.01 / 0.00
- kv17 @ 16M: – / 0.13 / 0.05 / 0.01
- kv17 @ 32M: – / 0.10 / 0.04 / 0.02
- kv17 @ 64M: – / 0.19 / 0.05 / 0.02
- kv17 @ 160M: – / 0.11 / 0.03 / 0.02
- kv17 @ 320M: – / 0.09 / 0.03 / 0.02
- kv33 @ 16M: – / 0.30 / 0.09 / 0.03
- kv33 @ 32M: – / 0.30 / 0.11 / 0.03
- kv33 @ 64M: – / 0.45 / 0.19 / 0.04
- kv33 @ 160M: – / 0.43 / 0.14 / 0.03
- kv33 @ 320M: – / 0.56 / 0.25 / 0.05
- kvb17 @ 16M: – / 0.07 / 0.03 / 0.01
- kvb33 @ 16M: – / 0.28 / 0.09 / 0.03
- kvb33 @ 32M: – / 0.36 / 0.12 / 0.03
- kvb33 @ 64M: – / 0.50 / 0.19 / 0.05
- kvb33 @ 160M: – / 0.73 / 0.30 / 0.08

## Fitted scaling trends


Primary law: Hill f1 = fmax x^g/(x^g+K^g) (the prior dense campaigns' form, debug/taskscale_lengthmix); secondary: saturating power law f1 = A - B x^-alpha. x = actual training PFLOPs. Target f1 per task: {'contradiction': 0.85, 'outlier': 0.45, 'nq': 0.88, 'oolong': 0.68}. With 4-5 points per curve a 3-parameter fit interpolates; treat targets beyond the largest measured budget as extrapolations.


## contradiction

| arm | points | best f1 (PF) | Hill fmax | g | K (PF) | rmse | PF to f1=0.85 (Hill) | x dense | satpow A / alpha | PF to target (satpow) |
|---|---|---|---|---|---|---|---|---|---|---|
| dense | 3 | 0.944 (1347.2) | 0.977 | 1.46 | 136.1 | 0.000 | 500.3 | 1.00 | 1.000 / 0.93 | 531.3 |
| ffnmoe-a10 | 1 | 0.050 (260.6) | - | - | - | - | - | - | - / - | - |
| ffnmoe-s1 | 3 | 0.910 (939.0) | 0.955 | 2.37 | 264.8 | 0.000 | 640.6 | 1.28 | 1.000 / 0.84 | 1045.3 |
| ffnmoe-s2 | 3 | 0.001 (1532.7) | 0.050 | 0.91 | 152763.6 | 0.000 | - | - | 0.001 / 1.34 | - |
| ffnmoe-t10 | 3 | 0.880 (998.1) | 0.898 | 2.58 | 219.4 | 0.000 | 670.1 | 1.34 | 1.000 / 0.86 | 843.4 |
| kv17 | 3 | 0.764 (429.2) | 1.050 | 1.19 | 189.0 | 0.001 | 637.3 | 1.27 | 1.000 / 0.66 | 983.0 |
| kv33 | 3 | 0.861 (694.9) | 0.899 | 1.94 | 139.1 | 0.000 | 604.0 | 1.21 | 0.993 / 0.92 | 598.9 |
| kvb17 | 1 | 0.053 (85.9) | - | - | - | - | - | - | - / - | - |
| kvb33 | 3 | 0.761 (637.6) | 0.833 | 2.16 | 214.0 | 0.000 | - | - | 1.000 / 0.77 | 1184.3 |

## nq

| arm | points | best f1 (PF) | Hill fmax | g | K (PF) | rmse | PF to f1=0.88 (Hill) | x dense | satpow A / alpha | PF to target (satpow) |
|---|---|---|---|---|---|---|---|---|---|---|
| dense | 5 | 0.922 (1138.6) | 0.962 | 0.57 | 5.2 | 0.005 | 336.2 | 1.00 | 0.970 / 0.47 | 336.0 |
| ffnmoe-a10 | 1 | 0.444 (269.6) | - | - | - | - | - | - | - / - | - |
| ffnmoe-s1 | 3 | 0.879 (780.4) | 0.919 | 1.67 | 122.8 | 0.000 | 790.7 | 2.35 | 0.972 / 0.95 | 763.6 |
| ffnmoe-s2 | 3 | 0.607 (1277.8) | 0.679 | 4.00 | 680.1 | 0.039 | - | - | 1.000 / 0.67 | 8606.2 |
| ffnmoe-t10 | 3 | 0.867 (848.2) | 0.870 | 3.98 | 217.1 | 0.000 | - | - | 0.999 / 0.90 | 838.3 |
| kv17 | 3 | 0.794 (205.0) | 0.779 | 4.00 | 73.3 | 0.022 | - | - | 0.849 / 1.16 | - |
| kv33 | 3 | 0.843 (351.7) | 0.839 | 4.00 | 108.8 | 0.009 | - | - | 0.893 / 1.11 | 1645.6 |
| kvb17 | 1 | 0.728 (88.5) | - | - | - | - | - | - | - / - | - |
| kvb33 | 3 | 0.878 (508.8) | 0.942 | 0.42 | 1.6 | 0.013 | 844.3 | 2.51 | 1.000 / 0.22 | 858.2 |

## oolong

| arm | points | best f1 (PF) | Hill fmax | g | K (PF) | rmse | PF to f1=0.68 (Hill) | x dense | satpow A / alpha | PF to target (satpow) |
|---|---|---|---|---|---|---|---|---|---|---|
| dense | 4 | 0.723 (1946.2) | 0.799 | 0.48 | 17.7 | 0.003 | 656.2 | 1.00 | 0.826 / 0.33 | 657.3 |
| ffnmoe-a10 | 1 | 0.508 (336.2) | - | - | - | - | - | - | - / - | - |
| ffnmoe-s1 | 3 | 0.679 (1316.8) | 0.690 | 1.76 | 127.1 | 0.000 | 1392.1 | 2.12 | 0.713 / 1.03 | 1245.4 |
| ffnmoe-s2 | 3 | 0.648 (2174.5) | 0.688 | 1.23 | 227.9 | 0.000 | 8215.5 | 12.52 | 0.705 / 0.88 | 5591.0 |
| ffnmoe-t10 | 3 | 0.688 (1436.8) | 1.050 | 0.28 | 145.0 | 0.000 | 1279.7 | 1.95 | 1.000 / 0.19 | 1283.2 |
| kv17 | 3 | 0.665 (564.1) | 1.034 | 0.10 | 1.3 | 0.003 | 917.0 | 1.40 | 0.726 / 0.26 | 1702.5 |
| kv33 | 2 | 0.663 (456.9) | - | - | - | - | - | - | - / - | - |

## outlier

| arm | points | best f1 (PF) | Hill fmax | g | K (PF) | rmse | PF to f1=0.45 (Hill) | x dense | satpow A / alpha | PF to target (satpow) |
|---|---|---|---|---|---|---|---|---|---|---|
| dense | 5 | 0.604 (3804.6) | 1.050 | 0.60 | 2516.7 | 0.012 | 1554.5 | 1.00 | 1.000 / 0.21 | 1424.5 |
| ffnmoe-a10 | 1 | 0.031 (271.0) | - | - | - | - | - | - | - / - | - |
| ffnmoe-s1 | 5 | 0.493 (5260.7) | 0.525 | 1.57 | 1023.6 | 0.007 | 3216.3 | 2.07 | 1.000 / 0.21 | 3726.2 |
| ffnmoe-s2 | 4 | 0.322 (4271.0) | 0.453 | 1.55 | 2397.5 | 0.009 | 67702.2 | 43.55 | 1.000 / 0.16 | 20489.6 |
| ffnmoe-t10 | 4 | 0.378 (2786.7) | 0.380 | 2.45 | 527.6 | 0.006 | - | - | 0.436 / 0.85 | - |
| kv17 | 5 | 0.089 (508.0) | 0.094 | 0.10 | 1.3 | 0.016 | - | - | 0.089 / 0.01 | - |
| kv33 | 5 | 0.284 (4013.2) | 1.050 | 0.28 | 179976.1 | 0.024 | 63716.3 | 40.99 | 1.000 / 0.06 | 740572.1 |
| kvb17 | 1 | 0.033 (95.5) | - | - | - | - | - | - | - / - | - |
| kvb33 | 4 | 0.369 (1753.2) | 1.050 | 0.59 | 4987.6 | 0.007 | 3065.4 | 1.97 | 1.000 / 0.13 | 6166.0 |


## Plots

- `/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/results/flop_scaling/contradiction_flop_fit35.png`
- `/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/results/flop_scaling/nq_flop_fit35.png`
- `/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/results/flop_scaling/oolong_flop_fit35.png`
- `/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/results/flop_scaling/outlier_flop_fit35.png`
