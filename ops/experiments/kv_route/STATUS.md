---
experiment: kv_route
title: Learned compute allocation — per-layer KV keep router, FFN-width router, block skip, joint (flexcompute) FLOP budget
status: done
updated: 2026-09-09
branch: prasann/landmark
launcher: research/ffn_compute/flop_scaling/orchestrate_scale.py
records:
  - records/kv-route-attention-allocation.md
results:
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/results/flop_scaling/results_scale_s4battn.csv
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/results/flop_scaling/results_scale_s4bflex.csv
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/results/flop_scaling/results_q3s4bdense2.csv
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/results/flop_scaling/results_q3s4bflex2.csv
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/results/flop_scaling/results_q3s4bflexs2.csv
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/results/flop_scaling/results_scale_s4bflexs.csv
  - research/kv_route/flexcompute_40x/README.md
  - research/kv_route/flexcompute_train/README.md
  - research/kv_route/flexcompute_train/LAUNCH_LEDGER.tsv
wandb_group: https://wandb.ai/prasann-uc-berkeley-electrical-engineering-computer-sciences/memory-networks/groups/flexcompute-train-g1
next_step: None queued; only untested lever is a long (>>31-step) trained run with a lower-bias gate init so block skip / KV keep can reach their targets — needs Prasann's go.
---

# kv_route

## Question
Can a model learn a task-dependent split of FFN compute AND attention/KV compute under one FLOP
budget (Prasann 2026-09-04: "router for cache allocation … highly flexible amounts of ffnn and
attention compute", hoping for ~60× combined), and is any such routed point compute-optimal vs dense
and/or a real wall-clock training speedup?

## Decisions
- 2026-09-04 (Prasann): test attention routing alone first (Stage A), then jointly with FFN (Stage B).
- 2026-09-04 (agent): pin all kvroute train AND eval to ai2/jupiter — FlexAttention on saturn A100 ran ~20× slower.
- 2026-09-05 (Prasann: "Qwen3 for now"): Stage C on Qwen3-4B (no GDN → larger routable share); third router = per-token block skip (`olmo_core.nn.block_skip`).
- 2026-09-05 (agent, ablation-proven): routers live on the model root and run on the block input OUTSIDE the AC region (`router_fn.RouterOnInput`) — anything inside a checkpointed block feeding the graph outside pins every block's activations (61–72 GB vs 27 GB).
- 2026-09-06 (agent): Stage D — block skip extended to GatedDeltaNet blocks (identity state update for skipped tokens).
- 2026-09-09 (agent): wall-clock recipe = head gating → static deletion of always-null FFN layers → COMPACTED (gather) block skip; KV router dropped as redundant once skip is compacted.
- 2026-09-09 (agent): trained-router ladder on real contradiction SFT (`research/kv_route/flexcompute_train`) — verdict: trained routers do not reach the pinned operating points in 31 steps, and forced ones break the task.

## Current best numbers
Mean f1 over 2k/8k/16k/32k, eval_size 500/rung (contradiction) / 500–600 (oolong) → ≈2000 per mean; no-cot; mult = dense FLOPs for same f1 ÷ arm FLOPs (>1 = compute-optimal).

| arm | task / rung | metric | eval_size | ±SE | cot | source |
|---|---|---|---|---|---|---|
| dense Qwen3.5-4B 56M | contradiction / mean | f1=0.944 @1347 PF | 4×500 | ±0.005 | no-cot | records/kv-route-attention-allocation.md (Stage B) |
| flex-c70 (joint, FFN 7×, keep 1.0) 56M | contradiction / mean | f1=0.927 @1020 PF, 0.757× FLOPs, **mult 1.15** | 4×500 | ±0.006 | no-cot | results/flop_scaling/results_scale_s4bflex.csv |
| attnroute-c50 (KV only, keep 0.5) 56M | contradiction / mean | f1=0.894, 0.955× FLOPs, mult 0.70 | 4×500 | ±0.007 | no-cot | results/flop_scaling/results_scale_s4battn.csv |
| attnroute-c25 / c10 56M | contradiction / mean | f1=0.005 / 0.002 (collapse below keep 0.5) | 4×500 | — | no-cot | results/flop_scaling/results_scale_s4battn.csv |
| flex-c70 80M | oolong / mean | f1=0.696 vs dense 0.723 (mult 0.73) | ≈2100 | ±0.010 | no-cot | results/flop_scaling/results_scale_s4bflex.csv |
| Qwen3-4B flex-c45 (2 routers) 80M | oolong / mean | f1=0.628 @0.63× FLOPs vs dense 0.667 (mult 1.39, steep dense curve) | ≈2100 | ±0.011 | no-cot | results/flop_scaling/results_q3s4bflex2.csv |
| Qwen3-4B flexs-c30 (3 routers) 56M | contradiction / mean | f1=0.000 (depth skipping kills retrieval) | 4×500 | — | no-cot | results/flop_scaling/results_q3s4bflexs2.csv |
| Qwen3.5-4B flexs-c15 (3 routers) 56M | contradiction / mean | f1=0.797 @0.57× FLOPs (mult 0.39) | 4×500 | ±0.009 | no-cot | records/kv-route-attention-allocation.md (Stage D) |
| fxt-g1-dense vs fxt-g1-moderate (trained, 31 steps) | contradiction_iid / 2k | f1=0.972 vs 0.190 | 500 | ±0.007 / ±0.018 | no-cot | /net/cubbins/data/prasann/flexcompute_train/eval/*.json (node-local) |
| pinned stacked routers (benchmark, no training) | throughput only | 5.63× @4B/16k, 5.72× @27B/8k; 3.90× packed | — | — | — | debug/flexcompute_40x/README.md |

**Verdict (2026-09-09):** one compute-optimal routed point exists (Qwen3.5-4B contradiction flex-c70, 1.15×, 2026-09-05), and it
is an FFN-only cut that a joint budget happened to anneal into — the joint router spends FFN first and cache last.
KV routing alone is never compute-optimal on the training axis (attention ≈12% of FLOPs <32k; ratio 0.92–0.955).
Three-router / block-skip arms lose to dense on every task and scale. Pinned wall-clock ceilings (5.6×) do not
survive training: in a 31-step SFT the gated routers never leave keep-all, and forcing capacity (2.8–7.4×) breaks
contradiction. Superseded as the FLOP-efficiency lead by soft-token KV (ds64) and drop-CPT (ffndrop).

## Next step
Nothing queued. If revisited: (a) gate init below the +10 keep-all bias or squared hinge / higher router LR so
short runs can reach budget; (b) per-layer KV keep floor (contradiction dies when L11+ caches empty);
(c) real fix for the eval `_compacted_block_mask` size mismatch (pad `Kp` to a FLEX_BLOCK_SIZE multiple).

## Open issues / traps
- Budget hinge under Adam: |mean−target| has constant gradient that Adam normalises; descent rate is set by router LR, so 14M/20M runs end near keep-all regardless of target — identical f1 across attnroute-c50/c25/c10 at 14M (0.7830) and c25/c10 at oolong 20M are that, not a copy bug. Treat short runs as near-dense controls.
- Block-skip checkpoints crash at eval in `kv_route._compacted_block_mask` (size 19 vs 20) when prompt length is not block-aligned; workaround `KV_ROUTE_BLOCKMASK=dense`. Not fixed as of 2026-09-09.
- Compacted block skip / KV-route / sparse-loss lm_head code was UNCOMMITTED on prasann/landmark until ported into the clean branch on 2026-10-12 (commits c5b495d39 / 686ea1488 on the flopdrop worktree branches) — memory:flopdrop-clean-branch.
- Qwen3 routed runs need 8 GPUs + the block-level BlockMask builder (`create_block_mask` materialised 8.6 GB/layer). Qwen3 eval config used Qwen3.5 eos 248044 → always 200 new tokens.
- Routed arms are SLOWER than dense at decode (mask build + per-layer sync); prefill break-even ~40k (2-router) / ~70k (3-router).
- Wall-clock: interleave arms in one process, dense control first and last — memory:gpu-benchmark-intraprocess-drift. FLOPs from `flop_meter/actual_pflops` — memory:flop-meter-actual-vs-total-petaflops. `flops.json` `*_frac` fields are run averages, not end operating points.
- Arm names must not start with `kv` (orchestrate35 routes kv* to the soft-token evaluator).

## Verified vs unverified
- Verified 2026-10-12: Stage A/B/C/D rows against the CSVs listed (flex-c70 0.9267 @1020 PF ratio 0.757; attnroute collapse; Qwen3 two/three-router rows); flexcompute_train ledger; trained-arm eval JSONs on cubbins (dense 0.972/0.904, headgate 0.977/0.904, moderate 0.190/0.021, cap arms 0.0 at 2k/8k); step seconds in frontier_main.json (dense 19.53 s, headgate 18.48 s, moderate 17.51 s, cap125 2.64 s).
- Read only in records/memory: Stage D per-rung numbers, pinned 5.63×/5.72× ceilings, eval-time decode slowdown.
- Conflict: flexcompute_train README says "the best speedup that preserves task quality is 1.15×", but frontier_main.json gives headgate (the only quality-preserving trained arm) 19.53/18.48 = 1.06×. UNVERIFIED — confirm with Prasann which number stands.
- Memory `compute-router-wallclock-findings` (2026-09-09) says "quality cost of skipping 75% … untested"; flexcompute_train (same day, later) tested it — forced skip at keep ≤1/4 diverges (CE 55–61) and keep 1/2 scores 0.0 at 2k. That memory is superseded on this point.
- Eval artifacts of flexcompute_train live only on cubbins node-local /data (not in repo results/). UNVERIFIED — confirm with Prasann whether they should be copied into results/.
