---
experiment: flop_scaling
title: FLOP-scaling study — dense vs routed FFN (ffnmoe) vs KV soft tokens (kv17/kv33), Qwen3.5 0.8B–27B, short-heavy 2k–32k
status: done
updated: 2026-09-08
branch: prasann/landmark
launcher: debug/flop_scaling/launch_grid35.py
records:
  - records/flop-scaling-ffn-kv-plan.md
  - records/flop-scaling-report-2026-09-02.md
  - records/kv-route-attention-allocation.md
  - records/ds64-handoff.md
results:
  - results/flop_scaling/results35.csv
  - results/flop_scaling/fits35.md
  - results/flop_scaling/scale_report.md
  - results/flop_scaling/results_scale.csv
  - results/flop_scaling/results_scale_s27b.csv
  - results/flop_scaling/ffn_speed_h100.md
  - results/flop_scaling/walltime.csv
  - debug/flop_scaling/LAUNCH_LEDGER.tsv
wandb_group:
next_step: None for this grid — KV follow-up lives in ds64, FFN follow-up in ffndrop; only open item is the unrun breadth-matched KV / 9B frozen-tail NaN, deprioritised.
---

# flop_scaling

## Question
At matched *training* FLOPs, is either nested-FFN routing (`ffnmoe`) or pooled-KV soft tokens
(fixed keep fraction: `kv17` = 1/6, `kv33` = 1/3) compute-optimal vs dense SFT on outlier / oolong /
contradiction / nq, short-heavy 2k–32k mixes, methods NOT combined (Prasann brief 2026-09-02)?

## Decisions
- 2026-09-01 (Prasann): pivot from Qwen3-4B to Qwen3.5-4B for everything; reuse the prior dense campaigns' points (`debug/taskscale_lengthmix`, `debug/outlier_lengthmix_scaling`) — Qwen3-4B grid cancelled.
- 2026-09-02 (Prasann): KV arms keep a fixed *fraction* of docs at every context, not a fixed count.
- 2026-09-02 (agent): FLOPs priced per example at real length; stage-2 FFN charged stage 1 + stage 2; all FFN numbers re-run after the meta-build random-router bug (fixed 545c07b0d).
- 2026-09-02 night (Prasann): phase 2 = (a) speed at fixed cost levels, (b) model-scale ladder 0.8B→27B (no dense 70B in family; 70B = speed-only geometry).
- 2026-09-03 (agent): 27B rung uses the POST-trained Qwen/Qwen3.5-27B (no -Base release) — every 27B number carries that caveat.
- 2026-09-04→05 (agent): FFN-only routing never compute-optimal → follow-up became the joint FFN+KV router (`kv_route` page).
- 2026-09-08 (agent/Prasann): KV line continued as the ds64 campaign with dense RETRAINED on new 2k–64k mixes; ds64 says never compare against fs35 dense points (different mix/shape).
- 2026-09-29 (Prasann): FFN line continued as drop-CPT (`ffndrop`), reusing `launch_grid35.py` with `FS35_BASE` override (commit 5c6645e77).

## Current best numbers
f1 = mean over the 4 eval rungs (2k/8k/16k/32k), native ladder evaluator, eval bundle `_eval_bundle_eval500_v3`; eval_size per rung 500 (contradiction) / 500–600 (oolong) / 600 (nq) → ≈2000–2400 per mean. PF = training PFLOPs at real lengths. SE = binomial approximation on the mean.

| arm | task / rung | metric | eval_size | ±SE | cot | source |
|---|---|---|---|---|---|---|
| dense 4B 56M | contradiction / mean | f1=0.944 @1347 PF | 4×500 | ±0.005 | no-cot | results/flop_scaling/results35.csv |
| kv33 (gold+1/3) 4B 56M | contradiction / mean | f1=0.861 @695 PF (mult 1.00) | 4×500 | ±0.008 | no-cot | results/flop_scaling/scale_report.md |
| kv33 9B 56M | contradiction / mean | f1=0.914 @1069 PF (mult 1.66) | 4×500 | ±0.006 | no-cot | results/flop_scaling/scale_report.md |
| dense 27B* 56M | contradiction / mean | f1=0.980 @7849 PF | 4×500 | ±0.003 | no-cot | results/flop_scaling/results_scale_s27b.csv |
| kv33 27B* 56M | contradiction / mean | f1=0.956 @3341 PF (mult 1.59) | 4×500 | ±0.005 | no-cot | results/flop_scaling/results_scale_s27b.csv |
| ffnmoe-t10 9B 56M | contradiction / mean | f1=0.905 @1746 PF (mult 0.91, best FFN) | 4×500 | ±0.007 | no-cot | results/flop_scaling/scale_report.md |
| kv17 (blind 1/6) 4B 20M | oolong / mean | f1=0.637 @134 PF vs dense 10M 0.624 @243 | ≈2100 | ±0.010 | no-cot | records/flop-scaling-report-2026-09-02.md |
| kv17 9B 80M | oolong / mean | f1=0.662 @816 PF (mult 0.75~, KV saturates) | ≈2100 | ±0.010 | no-cot | results/flop_scaling/scale_report.md |
| kvb33 (blind 1/3) 4B | nq / mean | f1=0.878 @509 PF vs dense-16M 0.878 @379 (dense wins) | 4×600 | ±0.007 | no-cot | records/flop-scaling-report-2026-09-02.md |

\* 27B = post-trained base, not comparable to base-model rungs. `~` = extrapolated beyond dense range.

**Verdict (2026-09-04, this grid):** routed FFN is never compute-optimal at any scale (0.47→0.91, falls to 0.67 at 27B);
KV soft tokens are task-shaped — compute-optimal on contradiction from 4B up (gold forced), win oolong only at
small scale/budget, trail on nq, dead on outlier.
**Superseded by later campaigns (do not quote this grid as the final word):**
- 2026-09-05 (kv_route): joint FFN+KV budget `flex-c70` = contradiction 0.927 @1020 PF, mult 1.15 — first routed point >1 at 4B.
- 2026-09-09→15 (ds64, dense retrained, 2k–64k): soft tokens DOMINATE measured dense points on all four tasks, e.g. contradiction `chk32-32M` 0.869 @639 PF vs dense-32M 0.868 @1539 PF (results/ds64/results.csv, 5 rungs), outlier `ck64-32M` 0.581 vs dense 0.564 at 4.0× fewer FLOPs (4-rung mean, records/ds64-handoff.md). The "outlier: nothing works" line of this grid is overturned.
- 2026-10-03 (ffndrop): routed t10 SFT from a drop-CPT base improves (contra 28M .808→.824, nq 32M .853→.879) but is still below the dense frontier.

## Next step
None for this grid. KV continuation = ds64 page; FFN continuation = ffndrop page; joint routers = kv_route page.
Never-run leftovers (deprioritised, not queued): breadth-matched KV (absolute 128–512 docs) on Qwen3.5;
all-layer FFN re-measure on fixed init; frozen-tail arm at 9B (two NaNs at steps 5/11, open bug).

## Open issues / traps
- The **2026-10-03 ledger rows are NOT this study**: 4 rows `fs35r2-{contradiction-…-s28M,nq-…-s32M}-bfdrop` (ffnmoe-t10 from the drop-CPT base) launched via this ledger by `launch_grid35.py`; first attempt LAUNCH-FAILED (`UnpushedChangesError`, another session's unpushed HEAD ec8242072), relaunched 04:26/04:27 as Beaker 01M40R8PBE5C192A0TFS7PE9NY / 01M40R9FN39TNRVGPJKXV3CM2S. Results belong to `ffndrop` (memory:ffndrop-cpt-task).
- FLOPs on compacted arms: read `flop_meter/actual_pflops`, never `throughput/total petaflops` (~10× wrong) — memory:flop-meter-actual-vs-total-petaflops.
- `--max-tokens` below one step's tokens silently gives a 1-step run that fakes the best matched-FLOP delta; filter `steps>=2` — memory:max-tokens-below-one-step-trap.
- Wall-clock: Berkeley H200 timing drifts up to ~2× inside one process; interleave arms, never compare across jobs — memory:gpu-benchmark-intraprocess-drift. Speed (A100) 1.1×→2.8× at 0.8B→70B-geom; H100 1.05×→2.2×.
- The 2026-09-02 claim "forcing gold leaks the answer on id-answer tasks" was confounded by the gold off-by-one bug (memory:gold-sidecar-index-base-bug); ds64 reversed it for nq.
- Soft-token on the hybrid must use `--attn-backend torch`; arm names starting `kv` are routed to the soft-token evaluator.

## Verified vs unverified
- Verified 2026-10-12 against primary files: scale_report.md table; results_scale_s27b.csv (27B dense/kv33 rows); eval_size 500/rung (contradiction 27B generations.jsonl), 600/500/500/500 (oolong 27B), 600/rung (nq dense-8M); ledger date histogram (rows only 2026-09-01→04 plus the 4 rows on 2026-10-03); ds64 rows for dense/chk32/ck64/occ00/kv33 in results/ds64/results.csv; git log (last flop_scaling-specific commit eaa53c589 2026-09-08; 5c6645e77 2026-09-29 is the ffndrop override).
- Read only in records/memory (not re-derived): 4B-grid numbers from the 2026-09-02 report (oolong kv17 vs dense, nq kvb33), speed numbers, ffndrop 2026-10-03 deltas, outlier ck64 4-rung means.
- Conflict: memory `flop-scaling-ffn-kv-study` says "ACTIVE (2026-09-02)"; evidence says the grid + scale ladder finished 2026-09-04 and only follow-on campaigns ran after. Marked `done`. UNVERIFIED — confirm with Prasann that nothing else (e.g. the 70B/27B speed or breadth-matched KV) is still expected from this key.
- wandb: launcher default group `flop-scaling-q35` in project `memory-networks`; entity not recorded, so no URL. UNVERIFIED — confirm with Prasann.
