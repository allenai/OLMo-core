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
