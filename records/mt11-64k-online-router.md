---
title: The mid-stack online token router on the 11-task CTC mix at 64k with ~64M tokens per task
date: 2026-10-23
updated: 2026-10-23
experiment: mt11_64k
area: token_routing
kind: result
status: interim (single seed at the 1/4-epoch point; seed replicates and 1/2 + full-epoch runs training)
superseded_by: 
artifact: https://claude.ai/artifact/5LcLKC8bT6MDfz6JcLLQ9n
summary: On an 11-task 64k mix (662M tokens), the otr10M7 online router at 0.40x dense FLOPs beats the better-LR dense curve on 9/10 tasks at matched LR 5e-6 (xabsence +0.125, outlier +0.051, contradiction +0.043; strmatch -0.018), gains growing with length; at LR 2e-6 it trails dense-5e-6 everywhere. Single seed, eval_size 500/cell.
---

# The mid-stack online token router at 64k, 11 tasks, ~64M tokens per task

Experiment state: `ops/experiments/mt11_64k/STATUS.md`. Code: branch `prasann/mt11-64k`,
`research/token_routing/mt11_64k/`. Page with the per-rung Pareto plots: the artifact above.

## Question

Prasann (2026-10-22): "for the 11-task mix try to go up to like 64k and have a larger amount of data (maybe similar
data to the individual task experiments from before for each task), since it seems like these methods maybe scale
better with more flops and longer contexts?" On the 8k fast8k 11-task mixes (480-720 rows/task) the router failed
strmatch (-0.12 to -0.25) and grouping (-0.05 to -0.07).

## Setup

- **Data `mt11L64`.** setA seed-42 gold shards (the generator the CTC suite was calibrated against; query position
  both; full gold sidecars), rows <= 65,536 tokens, ds64's short-heavy token shares 30/20/15/13/12/10 over
  <=3k/6k/12k/24k/48k/64k of a 64M-token target per task (`merge_tasks.py --length-mix`). 156,587 rows, 661.7M
  tokens, 12.3k-15.8k rows and 49-64M tokens per task. `verify_shard.py`: every row hits its gold sidecar; max row
  65,536. The top bin holds ~100 rows/task for 6 tasks, 12 for reorder, none for outlier/oolong/rerank (their 64k
  bucket rows exceed 65,536 tokens) or strmatch (max 20k).
- **Training.** Qwen3.5-4B markerfix base, 8 H100s, unpacked rows at seq 65,536, 128 rows/step, micro-batch 1,
  flash_2. Dense = the pad-free soft-token path with every document kept (`densepf`; on the mt4 8k screen it scored
  within noise of padded dense). Budgets f16/f8/f4/f2/f1 = 77/153/306/612/1,223 steps, each a complete LR schedule
  (`--max-tokens`).
- **Router `otr10M7`.** Blocks 0-6 on the full row; a ridge router refit every step on per-token gradient norms
  (10% exploration) keeps 10% of each document's tokens for blocks 7-31; a charged full-context step every 8; 512
  target samples per row.
- **Eval.** CTC-suite rows (HF `PrasannSinghal/ctc-suite-eval` at olmo-eval's pinned revision), 500 per rung,
  query position both, ctc-spec grading, native docchunk ladder on Beaker, no-cot. Rungs 2k/8k/16k/32k/64k where the
  suite has them (reorder to 16k; xabsence/strmatch/grouping to 32k). rerank is trained but not scored (no ladder
  grader for its suite rows). The 25 long cells were validated on all 500 rows before use (eval/train prefix parity
  500/500, gold scores 1.000, junk 0; longest prompt 70,035 tokens -> MAX_LENGTH 73,728, nothing skipped).
- **Comparison.** FLOPs = `flop_meter/actual_pflops`. Dense reference = per task x rung the better of LR 2e-6 and
  5e-6, interpolated linearly in log PF. No-signal cells excluded from means: reorder 8k/16k (every arm fails the
  exact-permutation parser: duplicate ids at 8k, repetition loops in 150/500 rows at 16k) and grouping 32k (every arm
  0.001-0.005).

## Results (single seed; eval_size 500 per cell; no-cot)

Cost: dense ~15 PF/step (f16 1,175 PF, f8 2,287 PF). Router f4 (306 steps): 1,872 PF actual vs 4,625 PF
dense-equivalent = 0.405x, i.e. between dense f16 and f8 (interpolated, not extrapolated).

Mean over each task's rungs of (router - dense at the router's PF):

| task | router LR 5e-6 vs best-LR dense | router LR 2e-6 vs best-LR dense | router LR 2e-6 vs dense LR 2e-6 |
|---|---|---|---|
| xabsence | **+0.125** (8k/16k/32k +0.14/+0.20/+0.15) | -0.015 | +0.082 |
| outlier | +0.051 (8k +0.11) | -0.051 | +0.056 |
| contradiction | +0.043 (16k/32k +0.07/+0.08) | -0.034 | +0.085 |
| oolong | +0.023 | -0.006 | +0.021 |
| qdmatch_nq | +0.022 | -0.014 | +0.060 |
| hotpotqa | +0.017 (32k/64k +0.04) | -0.009 | +0.032 |
| grouping | +0.016 (partial) | -0.027 | +0.020 |
| nq | +0.005 | -0.008 | +0.013 |
| strmatch | -0.018 | -0.090 | -0.041 |
| reorder (2k only) | +0.031 | -0.082 | +0.004 |

Binomial SE per cell is about ±0.01-0.02; run-to-run noise on the 8k screens was ±0.02-0.03 (up to ±0.13 on
outlier). Only the large long-rung wins (xabsence, contradiction 16k/32k, outlier 8k) clear a single seed; the rest
need the replicates.

- **The learning rate decides the sign.** At matched LR the router is ahead on 9/10 tasks; at 2e-6 it is ahead of
  2e-6 dense on 9/10 but behind 5e-6 dense on all. Dense 5e-6 beats dense 2e-6 at every measured budget on most cells.
- **Gains grow with length** on xabsence, contradiction, hotpotqa and qdmatch_nq; nq and 2k cells sit at parity.
- **strmatch**, the 8k screen's failure, is now within about one noise unit (-0.018) at matched LR.
- **Training on the mix fixes most of the length falloff:** nq f1 at 64k 0.17 (10-step smoke) -> 0.86 after 153
  dense steps.
- **No wall-clock saving** at micro-batch 1 on 8 GPUs: a routed step took 17-25 s vs ~20 s dense. The FLOP saving
  is real (0.40x) but this configuration is not compute-bound per micro-batch.
- Router health at 64k: keep_frac 0.105-0.108, leak_auc 0.47-0.52 (does not pick gold by count), gradient-mass
  fidelity at the full-context steps 0.62-0.66 (oracle 0.86-0.88, random 0.10).

## Traps hit

- Urgent Beaker jobs were preempted right after their 1 h minRuntime and restarted from scratch (no resume point
  before step 300): minRuntime now covers the whole run (capped at Beaker's 8 h) with a resume point every 50 steps.
- One node (`01JRYCX2HZQ2QE60PMHBB7ZM8J`) produced a NaN gradient at step 1; the same batch gives CE 0.9751 / grad norm
  26.82 elsewhere. Check the step-1 grad norm before suspecting the code.
- A routed step costs ~0.40x a dense step (not ~0.3x from the block count); the 10-step smoke's 0.52x included 2
  dense steps and overstated it.

## Open

Seed replicates (router f4, dense f8/f4, LR 5e-6) and the larger budgets (router + dense at f2 and f1, LR 5e-6) are
training; otr10M8 not yet run. This record is updated when they land.
