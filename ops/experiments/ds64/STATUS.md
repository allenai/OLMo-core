---
experiment: ds64
title: ds64 — soft-token (pooled-doc, detached slot) vs dense retrained, short-heavy 2k-64k data-scaling ladders, Qwen3.5-4B on Beaker
status: done
updated: 2026-09-22
branch: prasann/landmark
launcher: research/token_routing/ds64/launch_ds64.py
records:
  - records/ds64-handoff.md
  - records/ds64-scaling-plan.md
  - records/ds64-overnight-2026-09-14.md
  - records/trained-parity-winning-arms.md
  - records/outlier-realtoken-parity-probe.md
  - records/outlier-cc00-evaltime-probe.md
results:
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/results/ds64/results.csv
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/results/ds64/pareto.html
  - research/token_routing/ds64/LAUNCH_LEDGER.tsv
  - research/token_routing/ds64/ns_evals.tsv
  - research/token_routing/ds64_fast2k/
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/ds64_fast8k/results.csv
wandb_group: https://wandb.ai/prasanns-allen-institute-for-ai/memory-networks/groups/ds64-q35-4b
next_step: Resolve wall-clock before any new arm. Implement compact-then-pack (cu_seqlens, zero intra-micro-batch padding) for softtoken training; every Pareto claim is FLOPs-only until then
---

# ds64

## Question
On short-heavy 2k-64k mixes, is soft-KV (pooled-document) training Pareto-optimal against dense attention
retrained on the same data? Curves are accuracy vs tokens / FLOPs / wall-clock. Does the realised speedup
match the compaction ratio? (Prasann, 2026-09-08.) 4B first, then 27B.

## Decisions
- 2026-09-08 (Prasann): dense is RETRAINED on the ds64 mixes. Never compare to the old fs35 dense points (different mix and shape). Data, training and eval all run on Beaker.
- 2026-09-08 (agent): soft slots are fully detached (attention K/V plus the GDN write channels, `detach_soft_gdn`). Earlier KV arms had a gradient leak.
- 2026-09-13 (agent): RULE: compression wins where dense's scaling curve is flat. Predict the dense gain ratio before spending runs. Anchor cheap-end dense points: the extrapolated value was 0.07-0.15, the measured one 0.317.
- 2026-09-14 (agent): keep-mode rule.
  - nq and contradiction stay gold-FORCED.
  - Only outlier goes gold-blind: its gold doc's identity IS the answer.
  - The 09-02 opposite claim was confounded by the gold off-by-one (`memory:gold-sidecar-index-base-bug`).
- 2026-09-15 (agent): `--st-slot-mode cent_cmean` (content-only, row-centred mean) is free and makes the slot readable.
- 2026-09-15 (agent): first-k real body tokens per doc (`--st-header-extra-tokens K`) flips outlier to dominance. It generalises to contradiction but not to nq (kv33 stays the nq frontier).
- 2026-09-16 (agent): campaign CLOSED, all 153 runs and evals done. `first_last` (`--st-keep-token-rule first_last`) is the recommended default keep-token rule.
- 2026-09-22 (agent): slot-less arms (`--st-drop-slots`, chk32ns / flk64ns, commit 825e288f8) launched and evaluated. These are the last ds64 activity.

## Current best numbers
Mean f1 over 5 rungs (outlier: its 4 working rungs, 2k-32k). eval_size 500/rung. PF = `flop_meter/actual_pflops`.
SE is pooled over rungs (approximate).

| arm | task / rung | metric | eval_size | ±SE | cot | source |
|---|---|---|---|---|---|---|
| `occ00-16M` (keep 0, cent_cmean) 116 PF vs dense-16M 419 PF | oolong / 5-rung | f1 0.605 vs 0.603 (3.60x fewer FLOPs, parity) | 5x500 | 0.010 | unknown | results/ds64/results.csv |
| `kv33-64M` 556 PF vs dense-32M 1387 PF | nq / 5-rung | 0.878 vs 0.865 (2.49x, parity) | 5x500 | 0.007 | unknown | results/ds64/results.csv |
| `chk16-64M` 871 PF vs dense-64M 3056 PF | contradiction / 5-rung | 0.910 vs 0.922 (3.5x, near-parity −0.01) | 5x500 | 0.006 | unknown | results/ds64/results.csv |
| `chk64-16M` 349 PF vs dense-16M 758 PF | contradiction / 5-rung | 0.799 vs 0.763 (2.2x, dominates) | 5x500 | 0.008 | unknown | results/ds64/results.csv |
| `chk32-32M` 639 PF vs dense-32M 1539 PF | contradiction / 5-rung | 0.869 vs 0.868 (2.4x; 64k 0.656 vs 0.672) | 5x500 | 0.007 | unknown | results/ds64/results.csv |
| `ck64-32M` 396 PF vs dense-32M 1582 PF | outlier / 4-rung | 0.581 vs 0.564 (4.0x, dominates) | 4x500 | 0.011 | unknown | results/ds64/results.csv |
| `flk64-64M` 810 PF vs dense interp. 0.409 | outlier / 4-rung | 0.729 (+0.32, largest matched-FLOP delta; interpolated) | 4x500 | 0.010 | unknown | results/ds64/results.csv + handoff §1 |
| `chk32ns` / `flk64ns` (slot-less, `--st-drop-slots`) vs slotted `chk32` / `flk64`, 16M & 32M | contradiction 5-rung / outlier 4-rung | contra 0.747 vs 0.728 (16M), 0.863 vs 0.869 (32M); outlier 0.408 vs 0.426 (16M), 0.627 vs 0.612 (32M): slots add nothing in training (recorded 2026-10-21 from Beaker eval logs) | 500/rung (outlier 64k: ⚠ 300, excluded) | ~0.01 | no-cot | ops result (ds64, *ns*) |
| WALL-CLOCK: soft frontier vs dense point it "beats" | train_seconds / gpu_hours | chk32-32M 5044 s on 4 GPUs (5.6 GPU-h) vs dense 621 s on 8 GPUs (1.4 GPU-h); occ00-16M 3.50 vs 0.36 GPU-h; kv33-64M 5.25 vs 1.24; ck64-32M 2.33 vs 1.42 | — | — | — | results/ds64/results.csv |

## Next step
1. **Wall-clock (open risk).** Every frontier arm costs 1.6-9.7x MORE GPU-hours than the dense point it
   matches, at 0.25-0.42x the FLOPs. Length-sorted micro-batches (`microbatch_sort_pad_id`) recovered only 1.58x at micro 4.
   Compact-then-pack with `cu_seqlens` is not implemented: no commit since 09-16 touches it.
   The inference-side counterpart is solved (learned_router `fastskip`: 10x at 32k, batch 1).
2. Per-document adaptive budget, replacing fixed first-K. Its descendants are devloss_grid and learned_router.
3. 27B (`DS64_SCALE=27b`, titan B200) was never run, gated on item 1. No contradiction 128M anchor exists (ctc-data pool cap ~18k).

## Open issues / traps
- Outlier's 64k eval file (broken v2 xlong) reads ~0 for every arm. Quote 4 rungs, and never mix 4-rung and 5-rung means.
- Judge an arm by its best rung, not its mean. Check dominance over a MEASURED dense point before quoting an interpolated delta.
- Header-real x `--st-gold-blind` on an id-answer task (outlier `xhdr*`) collapses to the k/n chance floor. Qwen3.5 fuses `]:`, so outlier needs stop-id 5491, not 25. See `memory:header-real-pooling-parity`.
- The 900 s NCCL watchdog kills slow soft arms (oolong ohdr17, micro 4). Lower `ARM_MICRO`; do not raise the timeout.
- `--st-mix-*` was a silent no-op on gold-blind arms. It was fixed 09-14; the curriculum still does nothing on outlier.
- Packaging: ctc-data must be installed from `origin/prasann/ctc_public:ctc/` in the job. The top train rung is 56k (65536 window) while eval scores 64k. The orchestrator picks up `soft_arms.json` live, but not `launch_ds64.py`.
- Trained-parity: the chk32-32M and ck64-32M own-construction scores sit slightly below full text (dF1 −0.01 to −0.12, widening with length) (`records/trained-parity-winning-arms.md`).
- Read FLOPs per `memory:flop-meter-actual-vs-total-petaflops`. A run with `--max-tokens` below one step is a fake 1-step run (`memory:max-tokens-below-one-step-trap`).

## Verified vs unverified
- Verified 2026-10-12 against `results/ds64/results.csv` (213 rows, all scale=4b, last written 09-16 06:26):
  - every frontier row and dense anchor above;
  - the outlier 4-rung means (recomputed);
  - train_seconds, gpus and gpu_hours for the wall-clock row.
- Verified: `research/token_routing/ds64/ns_evals.tsv` lists 4 slot-less evals (flk64ns 16M/32M, chk32ns 16M/32M), all submitted 2026-09-22, with `ALL_DONE` in `launch_ns_evals.log`.
- UNVERIFIED: the slot-less results. Memory says outlier ns ≈ slot twin (16M .327 vs .344, 32M .503 vs .491). The contradiction ns scores are recorded nowhere in the repo, and `results.csv` predates them.
- UNVERIFIED (cot label): the ladder eval drivers do not state a cot mode.
- Caveat: train_seconds compares jobs on different GPU counts. Cross-job drift is up to 1.9x (`memory:gpu-benchmark-intra-process-drift`), so treat the GPU-hour ratios as order-of-magnitude.
