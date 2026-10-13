---
experiment: outlier_lengthmix
title: Length-mix data-scaling laws, dense vs sparse-landmark Qwen3.5-4B — outlier first, then 10 CTC tasks (taskscale)
status: paused
updated: 2026-09-09
branch: prasann/landmark
launcher: recipes/train/research_trainer/beaker_ctc_suite.py
records:
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/outlier_lengthmix_scaling/PLAN.md
results:
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/outlier_lengthmix_scaling/RESULTS_SNAPSHOT.md
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/taskscale_lengthmix/RESULTS.md
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/taskscale_lengthmix/fits.txt
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/taskscale_lengthmix/points.json
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/length_mix_scaling/STATUS.md
wandb_group:
next_step: Decide whether to run the built-but-unscored 128k ρ-range transfer test (n=880 arms, debug/outlier_lengthmix_scaling/build_128k.sbatch) or close the campaign at its 2026-09-09 verdict
---

# outlier_lengthmix

There is no `records/` writeup. The campaign's own result logs are the record:
`debug/outlier_lengthmix_scaling/RESULTS_SNAPSHOT.md` (2026-08-27 to 09-01) and
`debug/taskscale_lengthmix/RESULTS.md` (09-01 to 09-09). Plots are in `visualizations/outlier_lengthmix/`.

## Question
How much long-context SFT data does a task need, as a function of context length L and the length
mix? Does a sparse-landmark model ever overtake dense on tokens or wall-clock?

## Decisions
- 2026-08-27 (agent): per-arch LRs are full 5e-6 and sparse 1e-5 (check #1). Arms come from `compose_arms.py` (nested prefixes, SHUFFLE_SEED 7113) and train at seq 65536 via `beaker_ctc_suite.py`.
- 2026-08-28 (Prasann): DIRECTIVE. Prioritise the short-heavy mix (45/27/16/8/4% of tokens over 2k-32k). Don't spend runs on uniform, long-heavy or two-point mixes.
- 2026-08-29 (agent): dense `mixs64k96M` is ABANDONED. It hit NaN at step ~128 twice, from a pathological packed window. Dense `--pack` has no cu_doc_lens masking.
- 2026-09-01 (agent): extend to the CTC suite (`debug/taskscale_lengthmix/`), all on Beaker. rerank and qdmatch_hpqa are dropped.
- 2026-09-03 (agent): sparse fits with g at the bound or rmse > .05 are flagged `!STEP-NOT-HILL` and their crossovers suppressed. Extrapolated crossovers are treated as unconstrained-ceiling artifacts.
- 2026-09-07 (agent): short-heavy vs uniform is a sparse BOOTSTRAP, not a better mix. Dense ties (+.002 on the mean).
- 2026-09-08 (agent): practical rule. A single-seed sparse number is quotable only at its floor or well above takeoff. In the transition band it needs replicates, with the takeoff fraction reported.

## Current best numbers
Qwen3.5-4B, chat format, query_position=after. One seed unless stated.

| arm | task / rung | metric | eval_size | ±SE | cot | source |
|---|---|---|---|---|---|---|
| dense pure-8k, 32k ex | outlier / 8k | f1 0.980 | 600 | ±0.006 | unknown | RESULTS_SNAPSHOT.md |
| sparse pure-8k, 64k ex | outlier / 8k | f1 0.967 | 600 | ±0.007 | unknown | RESULTS_SNAPSHOT.md |
| dense / sparse short-heavy 640M | outlier / 16k | f1 0.894 / 0.816 | 600 | ±0.013 / ±0.016 | unknown | RESULTS_SNAPSHOT.md |
| dense / sparse short-heavy 640M | outlier / 32k | f1 0.532 / 0.386 | 600 | ±0.020 / ±0.020 | unknown | RESULTS_SNAPSHOT.md |
| dense pure-32k, 32k ex | qdmatch_nq / 32k | pair-F1 0.812 (asymptote ~.83) | 600 | ±0.016 | unknown | RESULTS_SNAPSHOT.md |
| dense / sparse 48M (seed 1) | nq / 32k | 0.873 / 0.248 (sparse seed 3: 0.545) | 600 | ±0.014 / ±0.018 | unknown | taskscale RESULTS.md |
| dense / sparse 160M | oolong / 32k | 0.652 / 0.554 | 500 | ±0.021 / ±0.022 | unknown | taskscale RESULTS.md |
| dense 56M / sparse 14M+28M | contradiction / 2k,8k,16k,32k | dense .982/.971/.941/.881; sparse .001/.000/.000 (dead at both budgets) | 500 | ±0.006-0.014 | unknown | taskscale RESULTS.md |

**Verdicts.**
1. Dense data need follows N(0.9) ≈ C_task·L^β with β ≈ 1-1.5 for every real task. Outlier is 8.7k/29k/70k examples at 8k/16k/32k. C spans ~15-50x across tasks. An apparent β=0 was saturation or the qdmatch lexical shortcut.
2. Sparse never beats dense on tokens at any rung or budget through 640M. Six of nine sparse arms sit at or near the task floor (contradiction, reorder, qdmatch dead). On outlier, sparse reaches wall-clock parity with dense at 16-32k (2-seed mean), not a win.
3. Sparse near its takeoff is bimodal across seeds. nq@48M took off in 3 of 4 seeds, qdmatch@160M in 1 of 4. Dense moves ≤.018 across seeds.

## Next step
The 128k transfer test (the "load-bearing extrapolation check") had its pilot measured: n=880 ≈ 128,335
tok/example (job 3483938). Arms were composed (`build_128k.sbatch`, `chaser_state/k128_repush.ids`). No
graded result is recorded anywhere found. Decide whether to train and score it. Also still owed: a
seq-65536-vs-16384 batch-confound control and an anchored-tail probe (memory, 2026-08-28).

## Open issues / traps
- Dense `--pack` has no cu_doc_lens masking. A 64k example packed with short ones NaN'd twice (memory:fsdp2-cp-ac-compile-checkpointerror).
- Never use `--master_port=0` for sparse local evals (torchrun waits 3x600 s, then dies).
- Sparse grouping 32k evals hang from an NCCL rank-imbalance desync. Use `--ngpu 1`.
- The q32k pool extension put the old 300-example heldout inside the new train prefix. Heldout CE for `lmx-full-q32k32000-qd-4b` is contaminated. The graded rungs are unaffected.
- `grouping`'s pairwise_f1 floor is ~0.44 (an all-one-cluster answer). Quote it with ARI.
- reorder's 60M budget is not buildable (2k pool tops out at 18,973 examples).
- Mix shares must come from measured rung medians, never rung labels (memory:ctc-rung-labels-not-tokens).

## Verified vs unverified
- Verified 2026-10-12: every table number against `RESULTS_SNAPSHOT.md` (mtime 2026-09-01) and `taskscale_lengthmix/RESULTS.md` (latest dated section 2026-09-09; mtime 2026-09-14).
- Verified: the 128k pilot measurements quoted in the `build_128k.sbatch` header.
- UNVERIFIED — confirm with Prasann: whether the 128k ρ-range arms were ever trained or evaluated. `debug/outlier_lengthmix_scaling/` has scripts dated 2026-09-24/25 (`build_128k`, `pilot_n880`, `build_nq_lengthmix`, `build_qdmatch_lengthmix`). They carry no result and may be mtime-only touches from a file move. Git shows no commits to these dirs after 2026-09-10.
- Memory `outlier-lengthmix-scaling-laws` (2026-08-28, "in flight: s96M/s160M + slm 64M seed-2") is stale. Those landed and are folded into the files above.
