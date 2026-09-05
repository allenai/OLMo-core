# Learned KV-cache allocation ("KV routing") — design, brainstorm, and experiment plan

*Started 2026-09-04. Code: `src/olmo_core/nn/attention/kv_route.py`, `src/olmo_core/nn/joint_budget.py`,
trainer variants `kvroute` / `flexcompute` in `train_ctc_suite.py`, arms `attnroute-c*` / `flex-c*`
in `debug/flop_scaling/launch_grid35.py`. Sibling of the nested-width FFN router
(`records/flop-scaling-ffn-kv-plan.md` §10–11).*

## Goal

A model that, after a task fine-tune, spends a **task-dependent** amount of FFN compute *and* a
task-dependent amount of attention/KV compute — both learned under a FLOP budget, not hand-set.
The FFN half exists (nested-width router). This adds the attention half and a joint budget.

## Brainstorm (five designs; 1 is implemented, 5 is next)

1. **Per-layer keep/drop (implemented).** Each full-attention layer has a `Linear(d_model,1)`
   router; a token's K/V is either written to that layer's cache (attendable by later queries) or
   dropped (the token still attends *from* itself). Tiers emerge: a token kept in *k* of *R*
   routed layers has cost *k/R*. Straight-through K/V scaling `1+p_sel−p_sel.detach()`, keep-all
   init (bias +10), two-sided budget on the mean keep probability, annealed target.
2. **Shared monotone tier.** One router per token emits tier *t ∈ {0..R}*; the token is kept in
   the last *t* routed layers. Fewer decisions, explicit tier vocabulary, but forbids "early layers
   need it, late don't".
3. **Per-head keep/drop (DMC-style).** Router per KV head; finer, but the cache saving only
   materialises if the kernel can handle ragged per-head lengths (Nawrot et al. 2024 do this with
   merge-not-drop).
4. **Merge into a running summary instead of dropping.** Dropped tokens fold into the previous
   kept slot (DMC). Preserves information; less clean as "compute allocation" and needs a custom
   kernel for the running average.
5. **Joint FFN+attention budget (implemented as `flexcompute`, not yet run).** Both routers on,
   *one* target on total FLOPs with the routers' FLOP shares as weights
   (`joint_budget.py`), so the fine-tune decides the split.

Design 1 was chosen because it is the exact analogue of the FFN router (same ST/budget
machinery, same "base is reproduced at init" property), needs no new kernel, and yields real
inference savings (cache compaction) with no per-head raggedness.

## Implementation notes

- **Training kernel.** FlexAttention over *compacted* keys: kept K/V gathered in position order
  (padded to a multiple of 1024) and concatenated with the full key set whose only allowed
  entries are the diagonal for dropped queries. Block mask (compiled `create_block_mask`) then
  skips everything outside the compacted causal frontier, so kernel work ∝ keep fraction.
  Measured per layer at 4B geometry (32q/8kv/128, 64k as 8×8k, fwd+bwd, H200):
  flash varlen 48.6 ms; routed keep-all 89.6 ms; keep 0.25 66.0 ms; keep 0.05 51.8 ms.
  The ~25 ms fixed overhead is `create_block_mask` evaluating the mask on the full grid; a
  block-level mask builder would remove it (`BlockMask.from_kv_blocks`). ≈5% of a 4B step, so
  left for later.
- **Inference.** Prefill runs the routed mask and then compacts each row's kept K/V into the
  KV cache right-aligned (leftpad += evicted count, `cache_seqlens = T`). Decode steps use the
  plain flash cached path (generated tokens always kept). Verified: cached prefill + 1 decode
  step == no-cache routed forward over the same tokens (rel 7e-3 bf16).
- **Cost accounting.** FLOP meter subtracts `(1−keep) × Σ_routed (attn.nfpt(L) − attn.nfpt(0))`
  (the length-dependent QKᵀ/PV part only; K/V projections for dropped tokens are *not* credited,
  so the number is conservative). `flops.json` carries `kv_route_keep_frac` and `ffn_cost_frac`;
  `collect_results35.arm_flops(lens, ffn_cost, attn_keep)` re-prices at real example lengths.
- **Eval.** `post_build_hook_from_config` (nested_ffn_moe.py) also enables `kv_route` from the
  exported `config.json`, so the docchunk/native evaluators score with the router.
- Tests: `src/test/nn/attention/kv_route_test.py` (CPU), `debug/flop_scaling/smoke_kv_route_gpu.py`.

## Experiment plan (4B, Qwen3.5, jupiter, FLOP-scaling grid recipe)

Stage A — attention only (`attnroute-c50/c25/c10` = keep targets 0.50/0.25/0.10), oolong
(20M/80M) and contradiction (14M/56M): 12 runs, orchestrator `FS_SCALE=4b FS_TAG_SUFFIX=attn`,
state `debug/flop_scaling/orchestrate_s4battn_state.json`, run names `fs35s4battn-<task>-attnroute-c<k>-s<B>`.
Stage B — combined (`flex-c*`: FFN router L12+ + KV router, joint target) after Stage A reads.

## What "60× overall" would need

At 32k context on 4B the per-token training FLOPs split roughly: FFN ≈ 57%, attention-score
(8 full-attn layers) ≈ 25–30%, fixed (Q/out projections, GDN layers, embeddings, LM head) ≈
13–18%. Even with FFN → 0 and attention keep → 0 the *total* saving is bounded by the fixed part:
≈6–8× on total FLOPs. 60× is reachable only on the *routed* share (FFN 100× × attention 20×) or
if the GDN layers and projections are also routed (a per-token "skip the whole block" rung —
design 6, not built). The reports quote both numbers: routed-share speedup and total.

## Run log

- 2026-09-04 22:10 Stage A launched (12 runs). Three oolong 20M runs landed on saturn A100s and ran
  ~20x slower than the H100 runs (TPS 258 vs 5.7k per device; 8 min/step) — FlexAttention on sm80
  is not viable here; cancelled and relaunched pinned to jupiter, evals pinned to jupiter too.
- Early routing (step 40–50, budgets on target): the router evicts from the TOP layers first.
  contradiction keep-0.5: L3/L7 1.00, L11 0.98, L15 0.62, L19 0.37, L23–L31 ≤0.02; oolong
  keep-0.25: L3–L11 0.47–0.70, L15 0.23, L19+ ≤0.02. The last full-attention layers give up their
  cache almost entirely at every budget; the first three keep most of it.
- The 14M contradiction runs (27 steps, anneal 8 steps) finished at keep ≈0.93 for all three
  targets — the anneal never reached them. They are near-dense controls, not budget points.
- Eval-time routing confirmed: every export's `config.json` carries the `kv_route` block (harvest
  2026-09-04 23:55), so `post_build_hook_from_config` enables the router before the strict load.
  (Eval logs hide INFO; both hooks now also print to stdout.)
- **Budget dynamics under Adam.** The three 14M contradiction runs and the three 20M oolong runs
  have byte-identical token-weighted keep fractions across targets 0.5/0.25/0.1 (0.9616 and
  0.8523). The two-sided |mean − target| term has a constant-magnitude gradient while the mean is
  above target, and Adam normalises it away, so the router descends at a rate fixed by its LR
  (1e-3) until it crosses a target; a 27-step run never gets there. The target anneal therefore
  does not pace eviction — the router LR does. Fix if short runs must hit budget: squared hinge or
  a higher router LR. Long-budget runs did converge onto target (0.51/0.25/0.09 hard keep).
- Final train CE (last 10 steps): contradiction 56M 0.039 / 0.53 / 0.67 at keep 0.5 / 0.25 / 0.1;
  oolong 80M 0.204 / 0.262 / 0.283. Contradiction breaks below half cache, oolong degrades
  gracefully; per-layer choice differs by task (contradiction spends a 10% budget on layer 3,
  oolong on layer 7).

## Stage A held-out results (4B, mean f1 over 2k/8k/16k/32k, eval_size 500 per rung)

| task / budget | dense | KV route 0.50 | KV route 0.25 | KV route 0.10 | soft-token ref | routed FFN t10 |
|---|---|---|---|---|---|---|
| oolong 80M | 0.723 | 0.671 | 0.643 | 0.625 | kv17 0.665 / kv33 0.675 | 0.688 |
| contradiction 56M | 0.944 | **0.894** (2k .969 / 8k .950 / 16k .888 / 32k .770) | **0.005** | **0.002** | kv33 0.861 | 0.880 |
| contradiction 14M (router stuck at keep 0.93) | 0.772 | 0.783 | 0.783 | 0.783 | kv33 0.525 | 0.579 |

- oolong degrades smoothly with cache size (−0.05 / −0.08 / −0.10), evenly across rungs; the
  keep-0.10 point sits near the soft-token 1/6 arm at a smaller cache.
- contradiction COLLAPSES below half cache: keep 0.25 and 0.10 score ~0 at every rung. Generations
  are well-formed pair lists (`[[3, 85], [47, 86], [65, 87]]` vs gold `[[3, 91], ...]`), i.e. the
  format is learned but retrieval is impossible once layers 11–31 hold ≤10% of keys. Consistent
  with train CE 0.53 / 0.67 vs 0.04 dense. The router's choice to keep two whole early layers and
  empty the rest is what a mean-keep budget rewards, but it removes the late-layer retrieval the
  task needs — a per-layer floor or a cost that penalises emptying a layer entirely is the obvious
  next knob.
- the 14M controls (keep 0.93) match/slightly beat dense (0.783 vs 0.772, within ±0.02 noise):
  evicting the 7% the router chose first is free.

### Compute axis (Stage A complete, 2026-09-05 01:00)

Priced at real example lengths (`results/flop_scaling/results_scale_s4battn.csv`), the *training*
FLOP saving of KV routing is small: ratio 0.955 (keep 0.5), 0.93 (0.25), 0.92 (0.10). The
attention-score share of training FLOPs on this short-heavy 2k–32k mix is only ~10–15% at 4B, so
even emptying the caches cannot buy much on that axis, and the matched-compute multipliers are all
< 1 (contradiction keep-0.5: dense reaches 0.894 at ~900 PF vs 1286 PF spent → 0.70x; oolong
keep-0.5 0.33x, keep-0.25 0.18x, keep-0.10 0.12x — oolong's dense curve is flat in FLOPs, so
every f1 point lost costs a lot of matched compute). On the training-FLOP axis KV routing therefore cannot be
compute-optimal at these lengths — the same conclusion the soft-token arms reached, and for the
same reason: attention is not where 4B training FLOPs go below 32k.

The axis where it pays is **inference**: a 50% smaller KV cache on every full-attention layer
(decode memory / bandwidth ∝ cache) for −0.05 mean f1 on contradiction (0.894 vs 0.944; still
above soft-token 1/3 at 0.861 and routed FFN at 0.880) and −0.05 on oolong; a 10x smaller cache
for −0.10 on oolong. Contradiction cannot go below half.

**Stage B (combined, `flex-c70` / `flex-c60`)** launched on the large budgets only: joint budget
with FLOP shares evaluated at 8k (`--flex-share-seq-len`), because shares at the 65k padded window
over-credit attention (score FLOPs ≈ 50% there vs ~12% at real lengths) and would steer the whole
saving into eviction. Achievable floor at 8k ≈ 1 − 0.36 (FFN L12+) − 0.12 (attn) ≈ 0.52, so 0.70
and 0.60 leave the router a real choice of split.

### Stage B (joint budget) — how the model splits the saving (2026-09-05 02:00, training-time)

Shares at 8k: FFN(L12+) 0.347, attention-score 0.131, fixed 0.522.

| run | joint target | FFN mean cost (L12+) | KV keep | final train CE |
|---|---|---|---|---|
| contradiction 56M flex-c70 | 0.70 | 0.145 (6.9x) | **0.997** | 0.033 |
| contradiction 56M flex-c60 | 0.60 | 0.008 (~120x) | 0.56 (L3–L11 full, L15 .63, L19–L27 ~.1, L31 .51) | 0.058 |
| oolong 80M flex-c70 | 0.70 | 0.141 (7.1x) | **0.998** | 0.196 |
| oolong 80M flex-c60 | 0.60 | 0.009 (112x) | 0.58 (L3–L15 full, L19 .45, L23+ ≤.1) | 0.260 |

Given a free choice, BOTH tasks spend the entire 30% saving on FFN width and keep every key
(keep 0.998 at target 0.70). Only when the budget exceeds what FFN can give (0.60 needs 0.40 >
0.347) does the router start evicting cache, and then it still drives FFN to ~1% first. So the
learned preference is FFN-first, cache-last — the opposite ordering from the Stage A intuition
that attention is the cheap thing to cut. Contradiction at 0.60 sits at CE 0.058 vs dense 0.04
with L12+ FFN at 1/120 width AND 44% of keys evicted; the eval will say whether that holds.
(Dense-priced: 0.60 of dense training FLOPs = 1.67x; on the ROUTED share it is FFN 120x x
attention 1.8x.)
