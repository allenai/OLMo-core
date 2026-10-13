---
experiment: learned_router
title: End-to-end learned token-drop router (+ per-(token,layer) skip router) on the frozen dense 4B CTC checkpoints, scored by answer-CE on the dev-loss grid
status: active
updated: 2026-10-02
branch: prasann/landmark
launcher: research/token_routing/learned_router/train_e2e_router.py
records:
  - records/learned-token-router.md
  - records/devloss-grid-underperformers.md
results:
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/learned_router/layerskip/frontier_summary.txt
  - research/token_routing/learned_router/layerskip/frontier.json
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/learned_router/layerskip/runs/nq/
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/learned_router/headline_shared.json
  - research/token_routing/learned_router/split_report_test64.json
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/devloss_grid/results_router_t64/
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/devloss_grid/results_router_long/
wandb_group: https://wandb.ai/prasann-uc-berkeley-electrical-engineering-computer-sciences/memory-networks/groups/router-e2e
next_step: Fix 2k->32k length transfer (calibrate the layer cutoff on long val rows or train on mixed lengths), then take the routed construction from eval-side CE to a trained/generation-graded check
---

# learned_router

## Question
Can a learned per-token keep/drop router replace the dev-loss grid's fixed rules (`gold_fl20p8_noslot`
and similar)? The router is trained end to end on the frozen dense Qwen3.5-4B suite checkpoints, with no
heuristic init. A second question is how far a per-(token, layer) skip router on top of it can cut forward
FLOPs at ≤0.02/0.05/0.10 nats ΔCE.

## Decisions
- 2026-09-23 (agent): v1 was a linear router over [embedding, position, gold flag] trained with REINFORCE. It was bistable: keep-all or collapse. It beat the heuristics only on nq/outlier/msmarco.
- 2026-09-28 (agent): a differentiable relaxed router with a task-relative tolerance (§7) did not pass the gate on nq/outlier/contradiction, so it was not run on the full grid.
- 2026-09-29 (Prasann): the router must be END-TO-END. Never warm-start from or distill fixed rules such as gold_fl20p8: "optimal rules should emerge naturally". A neutral init plus a keep-all fallback is fine. Standing directive: `memory:router-end-to-end-no-heuristic-init`.
- 2026-09-30 (agent): canonical recipe v5/v6a.
  - `relpos_noemb` features, ρ matched to the bar, hard-concrete gates, CE+KL, 3 restarts plus polish.
  - The primary comparison is the paired per-row budget (`@pair`) against `gold_fl20p8_noslot`.
- 2026-10-01 (agent): stage 2 is a per-(token, layer) skip router on the frozen token router (`layerskip/`). Frontier tiers are 0.02/0.05/0.10 nats.
- 2026-10-02 (agent): recommended shared router = seed-averaged 6-task `xs6avg` (`weights/xs6avg/xs6avg_s0_rhobar.pt`). 17-task pooling is high-variance: `xt17v6a`'s 1/13/2 was a lucky seed.

## Current best numbers
All are teacher-forced answer-CE deltas on frozen dense 4B checkpoints. Nothing here is generation-graded.
⚠ eval_size is <500 everywhere, by design of the grid.

| arm | task / rung | metric | eval_size | ±SE | cot | source |
|---|---|---|---|---|---|---|
| token v6a + layer-skip, ≤0.05 tier | 16 tasks / 2k | mean per-task FLOP cut 4.48x (pooled x0.412 = 2.43x); meets on test 12/16 | 64/task ⚠ eval_size=64 | per-task | no-cot (TF CE) | debug/learned_router/layerskip/frontier_summary.txt |
| bar `gold_fl20p8_noslot` | 16 tasks / 2k | FLOPs x0.473, mean ΔCE +0.098 | 64/task ⚠ | — | no-cot (TF CE) | same |
| token f0.35 + layer keep 0.5 | nq / 2k | 10.24x fewer FLOPs, ΔCE −0.0015 | 64 ⚠ | 0.004 | no-cot (TF CE) | same |
| token f0.35 + layer keep 0.5 | outlier / 2k | 7.28x, ΔCE +0.021 | 64 ⚠ | 0.012 | no-cot (TF CE) | same |
| per-task v6a token router @pair | 16 tasks / 2k | 5 beat / 9 match / 2 lose (niah, outlier_amzn) vs bar | 64/task ⚠ | paired | no-cot (TF CE) | records/learned-token-router.md §9 |
| shared xs6avg @pair | 16 tasks / 2k / 8k / 32k | 12/16 at parity at 2k; 8k 1/7/7, 32k 1/5/9 (beat/match/lose) | 64 (2k), 48-56 (long) ⚠ | paired | no-cot (TF CE) | records/learned-token-router.md iters 65, 67 |
| token 0.5x bar (x0.099 FLOPs) | nq / 32k | ΔCE vs bar −0.0014; 10.3x wall-clock (CUDA graph, batch 1) | 56 ⚠ eval_size=56 | 0.0009 | no-cot (TF CE) | debug/learned_router/layerskip/runs/nq/longcheck_32k_L0.5_k0.5.json |
| + layer keep 0.75 (x0.078) | nq / 32k | 12.3x wall-clock vs full; ΔCE vs bar +0.000 | 56 ⚠ | 0.002 | no-cot (TF CE) | runs/nq/fastskip_32k_b1_k0.75.json (timing only) |
| + layer keep 0.5 (x0.054) | nq / 32k | ΔCE vs bar +0.110 (22/56 rows bad): does not transfer | 56 ⚠ | 0.015 | no-cot (TF CE) | longcheck_32k_L0.5_k0.5.json |

## Next step
Length transfer is the main gap: 2k-trained routers lose most 32k cells, and layer keep 0.5 breaks at 32k.
Calibrate the layer cutoff on long val rows or train on mixed lengths. After that, check that the CE
parity holds under generation, in a trained model. Other open items from the record: niah needle cut on
some test64 rows, and textgroups at 32k needs doc-level features. The router weights (.pt) stay on local
storage and are not in git (`memory:flopdrop-clean-branch`).

## Open issues / traps
- Every verdict uses eval_size 16-64 rows/task. The record says to read spreads and medians, not third decimals. Run-to-run variance (GPU nondeterminism) changes verdicts on single seeds, so require ≥3 seeds for subset claims.
- CE parity does not imply a right answer (`memory:ce-vs-chance-floor`). No generation-graded check of a routed construction exists yet.
- The xabsence column is VOID: the paraphrase-era checkpoint is scored on EXACT rows. niah, fiqa and absence select on only 16 val rows (`_stg`).
- Pooled training with `--xtask-offload` was 80% model swaps; fixed by pinned-memory parking (iteration 60).
- Layer skip converts to wall-clock only with fixed-capacity gathers plus CUDA graphs (`fastskip.py`). At 2k batch 1 it is latency-bound (2.8x vs 3.0x for token-only).
- At 32k the full-context model is itself +0.065 ± 0.030 worse than the bar on nq.

## Verified vs unverified
- Verified 2026-10-12 from `layerskip/frontier_summary.txt`: the tier means (≤0.05: x0.412 pooled, 4.48x mean per-task, 12/16 meets), nq 10.24x, outlier 7.28x, and the bar x0.473 / +0.098.
- Verified from `layerskip/runs/nq/longcheck_32k_L0.5_k0.5.json`: full +0.065, token 0.5x bar at x0.099 −0.0014 ± 0.0009, and +layer keep 0.5 +0.110 ± 0.015 on 22 of 56 rows. `fastskip_32k_b1_k0.75.json` gives 12.27x graphed vs full at x0.078.
- UNVERIFIED: the quality of layer keep 0.75 at 32k (+0.000 ± 0.002). The record cites `runs/nq/longcheck_32k_L0.5_k0.75.json`, but no such file exists under `debug/learned_router/`; only `_k0.5` does.
- UNVERIFIED: the per-task 5/9/2 and xs6avg 12/16 tallies. They were read from the record's iteration log and not recomputed.
- UNVERIFIED — confirm with Prasann: is this still active on `prasann/flopdrop_clean`? There has been no commit since ec8242072 (2026-10-02).
