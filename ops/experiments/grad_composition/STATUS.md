---
experiment: grad_composition
title: Per-token x per-block gradient / activation composition of the answer loss over CTC fine-tuning (Qwen3.5-4B), to guide train-time token / layer router design
status: done
updated: 2026-10-23
branch: claude/grad-composition-6f09c7
launcher: research/token_routing/grad_composition/run_probe_lambda.sbatch (Lambda node02, 1 A100) and launch.py (Beaker jupiter, 1 H100)
records:
  - records/grad-composition.md
results:
  - research/token_routing/grad_composition/results/ (gitignored; <label>/probe.json + probe_stats.npz; Lambda copy /accounts/projects/sewonm/prasann/ctc_suite/gradcomp_out/)
artifacts:
  - https://claude.ai/artifact/BVaPjZdAKjETHdQDdGqqtQ
wandb_group:
next_step: v2 COMPLETE (2026-10-23; record + artifact current). Proposed follow-ups: grad_fidelity.py on strmatch / grouping rows with the trained router; a short-document token floor in the online router tested on mt11 strmatch; generation-level look at grouping (router vs dense have equal TF CE there).
---

# grad_composition

## Question
How does the answer-CE gradient (and activation-side signal) distribute over tokens and blocks of Qwen3.5-4B on
CTC 8k rows, and how does that change from the base through dense fine-tuning (mt4: 31 -> 366 steps; mt11: 165 -> 495)
and under the online router? Goal: where a train-time router should split, whether one keep decision can serve all
later layers, gradient vs activation targets, and why strmatch / grouping resist dropping.

## Decisions
- 2026-10-22 (parent session online-router / tokdrop_lr, brief): measure ||dL/dh_b,t|| and activation quantities for
  every token x block on held-out fast8k rows, over the dense matched-FLOP checkpoints + 1-2 router checkpoints;
  tasks at least contradiction, nq, outlier, oolong, strmatch, grouping, hotpotqa.
- 2026-10-22 (agent): one probe row set for every checkpoint = mt11s960 rows 720-751 per task (11 tasks): mt11
  shards are nested prefixes (s480 in s720 in s960), so rows >= 720 are unseen by every mt11 run up to s720, and mt11
  rows are disjoint from all mt4 shards (checked by md5 on Lambda; the probe re-checks and logs `seen_in_training`).
- 2026-10-22 (agent): block run as two checkpointed halves (mixer / FFN) to observe the residual stream inside each
  block; parameter-gradient share per token = ||delta_t|| * ||x_t|| at w_out and w2 (exact Frobenius norm of the
  token's rank-1 term); attention mass from _prepare_qkv q/k at the 8 full-attention blocks.
- 2026-10-23 (agent): Beaker jupiter backlogged (13 jobs pending ~1h, never started) -> cancelled, moved to Lambda node02
  where the checkpoints are staged; 64 rows/task (rows are cheap: ~1.5-3 s). mt4 rows = mt4s1952 rows 1464-1527 (mt4
  shards are true prefixes, verified). mt11 rows = mt11s720 rows 480-543 BUT mt11 shards are SHUFFLED supersets (42/64
  are mt11s480 rows): every mt11 comparison uses the 19-23 rows/task seen by neither mt11s240 nor mt11s480
  (results/mt11_seen.json). mt11 rows are setA format -> out of format for mt4-trained checkpoints (reference only).
- 2026-10-23 (parent + online-router sessions): asked first for (1) the blocks 0-6 vs 7-31 parameter-gradient share
  and (2) cross-layer rank correlation (L3/L7/L16/L24) for their early-nograd and pyramid-drop runs; sent at ~08:00 UTC.
- 2026-10-23 (agent): node02 then saturated by the online-router session's 2-GPU preemptive runs (11 queued); my
  normal-QOS probes were preempted twice. Did not take preemptive_high (would preempt that session's training);
  relaunched the remaining checkpoints on Beaker from the weka copies.
- 2026-10-23 (agent): Beaker wave (13 jobs, 10-40 min each) = PRIMARY series for v2: mt4 rows (dense LR 5e-6 31/61/122/366,
  dense LR 2e-6 366, router otr10M8 366) and mt11s960 rows 832-959 filtered to rows no mt11 shard contains (32-42/task;
  dense LR 2e-6 165/330/495, router otr10M8 330/495). Lambda pass = replication (base identical; dense-366 is a separate
  training run, within 0.03 / 0.08). LR 2e-6 control: the router's cross-layer stability is mostly the LR; its forward
  adaptation (+0.02 vs +0.20 dCE) and early-heavy gradient (0.31 vs 0.22) are router effects.
- 2026-10-23 (agent): key numbers recorded with `ops result add --check` (913 rows, metrics gradcomp_*; a duplicate add
  was removed under the results lock).

## Current best numbers
<!-- One line per number: task, rung, metric=value, eval_size, ±SE, cot label, source path.
     Flag eval_size<500 inline. Prefer `ops result show --experiment grad_composition --best`. -->
| arm | task / rung | metric | eval_size | ±SE | cot | source |
|---|---|---|---|---|---|---|
| base -> mt4 dense 31 / 122 / 366 | mt4 pooled, 8k | param-grad energy share blocks 0-6: 0.45 -> 0.31 / 0.25 / 0.23 (mt11: 0.45 -> 0.34 at 495) | 64/task ⚠ | ±0.01 | no-cot (TF grad) | ops results gradcomp_pgrad_share_blocks0_6 |
| base -> mt4 dense 31 / 122 / 366 | mt4 pooled, 8k | Spearman G7~G24 0.57 -> 0.47 / 0.41 / 0.34; doc share of G^2 at block 7 0.15 -> 0.54 / 0.55 / 0.57 | 64/task ⚠ | ±0.01-0.02 | no-cot (TF grad) | gradcomp_rho_G7_G24, gradcomp_docshare_G2_b7 |
| base / dense 366 / dense LR2e-6 / router | mt4 pooled, 8k | fwd ablation keep 10%/doc before block 7, dCE: gradient +0.13 / +0.23 / +0.20 / +0.02; activation +0.47 / +0.50 / +0.64 / +0.52; random +0.35 / +0.66 / +0.73 / +0.46 | 64/task ⚠ | ±0.02-0.05 | no-cot (TF CE) | gradcomp_abl_dce_keep10pct_b7_* |
| mt11 dense 495 vs router 495 | strmatch 8k | oracle gradient keep dCE +0.67 vs +0.08; TF CE 0.034 vs 0.111; doc share of G^2 at block 24 0.24 vs 0.39 | 37 ⚠ eval_size=37 | ±0.05-0.1 | no-cot | record findings 6 |
| mt11 dense 495 vs router 495 | grouping 8k | TF CE 0.325 vs 0.329 (no loss gap; router's eval deficit is generation-only) | 34 ⚠ eval_size=34 | ±0.03 | no-cot | record finding 7 |

## Next step
v2 is complete. Follow-ups (each its own run, not started): grad_fidelity.py on strmatch / grouping rows with the
trained router; a short-document token floor for the online router on mt11 strmatch; inspect grouping generations,
router vs dense.

## Open issues / traps
- mt11 shards (s240/s480/s720/s960) are SHUFFLED supersets, not prefixes: "rows past N" of a bigger shard are NOT held
  out from an sN run. probe.py now records per-row `seen_in`; filter on it. mt4 shards ARE prefixes.
- mt11 rows (setA format, query both sides) are out of format for mt4-trained checkpoints.
- Lambda node02 is shared with the online-router session's runs; normal-QOS jobs get preempted.

## Verified vs unverified
- Verified (how, when): probe end to end on Lambda (smoke 113921, 2026-10-23); seen-row checks by md5 on Lambda
  (results/mt{4,11}_seen.json); per-row stats consistent between all-64 and clean-subset mt11 rows (2026-10-23).
- UNVERIFIED — confirm with Prasann:
