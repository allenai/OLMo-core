---
experiment: nested_ffn_moe
title: Nested-width FFN mixture — learned per-token router over full/…/1-unit/null FFN rungs under a compute budget (standalone FFN axis)
status: done
updated: 2026-09-21
branch: prasann/landmark
launcher: research/ffn_compute/ffnmoe_train/run_q4b_ffnmoe.sbatch
records:
  - records/flop-scaling-ffn-kv-plan.md
  - records/flop-scaling-report-2026-09-02.md
  - records/ffndrop-cpt-plan.md
results:
  - research/ffn_compute/ffnmoe_train/LAUNCH_LEDGER.tsv
  - research/ffn_compute/ffnmoe_train/README.md
  - research/ffn_compute/ffnmoe_bench
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/results/flop_scaling/ffn_speed_h100.md
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/results/flop_scaling/scale_report.md
  - research/kv_route/flexcompute_40x/README.md
  - /accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/flexcompute_40x/ffn_moe_arm_4B_32768.json
wandb_group:
next_step: None in this family — the open FFN-routing question moved to ffndrop (drop-CPT base, then routed SFT); untried here: importance-permuting FFN units before nesting, 9B frozen-tail NaN.
---

# nested_ffn_moe

## Question
How little per-token FFN compute can a dense model get away with (target 10–100×) if a learned router
picks one of several NESTED FFN widths (MatFormer/MoNE style, null rung at the bottom) under a
budget hinge — with quality held at dense, and studied INDEPENDENTLY of KV/pooled-doc work?

## Decisions
- 2026-08-31 (Prasann): standalone FFN study — do NOT combine with pooled-doc-KV until later; plain causal attention.
- 2026-08-31 (agent): nested widths, not AdaMoE (AdaMoE = same-size experts + null experts on an existing MoE, ~14% saving; Qwen3-4B is dense) — memory:adamoe-vs-nested-experts.
- 2026-09-01 (Prasann): launcher default `--ac-mode none` ("no AC needed").
- 2026-09-02 (agent): quote batch-1 deployed costs only — batch-8 padded evals route pads to full and overstate cost ~0.05–0.08.
- 2026-09-02 (agent): every result before the meta-init fix 545c07b0d (router started RANDOM on meta-built models) discarded for fs35; v-series Qwen3-4B results kept (healed by long horizons).
- 2026-09-02 (agent): two-sided budget |cost−target| (t10/a10 arms) adopted for the FLOP-scaling grid; v12 recipe = two-stage + fine ladder.
- 2026-09-09 (agent): default `--ffn-moe-divisors` → `1,2.4,4.9,10,32` (width_multiple 8) — rungs below ~300 width buy FLOPs but no time; trained routers do layer pruning, so static deletion of always-null layers beats routing them (1.20× vs 1.18×).
- 2026-09-29 (Prasann): next attempt = drop-CPT then routed SFT (ffndrop), since routed FFN was never compute-optimal vs dense SFT.

## Current best numbers
Qwen3-4B, contradiction n=100 (≈5k tokens), held-out `contradiction_eval_pubmed_both_n100_k3`, eval_size=488 (the whole file; accepted as-is), batch-1 deployed routing cost, no-cot.

| arm | task / rung | metric | eval_size | ±SE | cot | source |
|---|---|---|---|---|---|---|
| q4b-ffnmoe-dense | contradiction n100 | f1=0.988 (2nd dense sample 0.982 = noise floor) | 488 | ±0.005 | no-cot | src/scripts/train/memexpress/ffnmoe/LAUNCH_LEDGER.tsv |
| v5 t05 L12–35 | contradiction n100 | f1=0.976 at 17× on routed layers (2.7× total FFN) | 488 | ±0.007 | no-cot | same ledger |
| v10 t01 L12–35, 7-rung ladder incl. width-1 | contradiction n100 | f1=0.973 at 127× on routed layers | 488 | ±0.007 | no-cot | same ledger |
| v9 two-stage, all 36 layers t05 | contradiction n100 | f1=0.939 at 20× all layers | 488 | ±0.011 | no-cot | same ledger |
| v12 two-stage + fine ladder t02 all layers | contradiction n100 | f1=0.929 at 40× all layers (only L0 dense) | 488 | ±0.012 | no-cot | same ledger |
| ffnmoe-t10 Qwen3.5-4B 56M (FLOP grid) | contradiction 2k–32k mean | f1=0.880, mult 0.81 vs dense (never >1 at any scale; best 0.91 @9B) | 4×500 | ±0.007 | no-cot | results/flop_scaling/scale_report.md |
| measured FFN speed, cost 1/16 (H100, train) | throughput | model-level 1.05× (0.8B) → 1.22× (4B) → 1.61× (27B) → 2.2× (70B geom) | — | — | — | results/flop_scaling/ffn_speed_h100.md (via plan §11) |
| null routing vs top-k MoE, 4B, T=32768 (H200) | FFN time | dense 26.88 ms; best full-or-null 1.35 ms (19.9×); best MoE top-2 1.90 ms (14.1×), top-8 6.54 ms | — | — | — | debug/flexcompute_40x/ffn_moe_arm_4B_32768.json (memory:moe-vs-null-routing-ffn-time) |

**Verdict:** on late layers the FFN is nearly free to cut (127× on L12–35 within ~2 SE of dense; all-layer
ceiling ≈0.94 at ≤0.05 budget in every variant). But it does not pay: FFN FLOP cuts cap total savings
(~0.70× at 4B lengths), never reach compute-optimality vs dense SFT (flop_scaling, 2026-09-04), and convert
poorly to wall-clock (1.05–1.2× at 4B). Standard top-k MoE floors 2.3–5.5× above null routing on FFN time
(2026-09-21), so MoE geometry is not the fix either.

## Next step
None here. FFN-routing continuation = `ffndrop` (2026-10-03: routed t10 SFT from drop-CPT base still below dense
frontier). Never-tried levers: importance-permute FFN hidden units before nesting (exact reparameterisation, free);
gentler anneal / `--ffn-moe-recon-weight` / exempt L0 from the budget mean.

## Open issues / traps
- OPEN BUG (2026-09-04): `--ffn-moe-trainable-width` (frozen tail, ffnmoe-t10p) NaNs at step 5–11 on 8-rank FSDP (9B, 2/2 runs); 27B single node OOMs at 75 GB. Pulled at 9B/27B.
- Routed runs need a base baked by `bake_ffn_moe_into_base.py` and must be scored with the same routing flags (eval sbatch refuses mismatch; flags live in checkpoint `config.json` `ffn_moe` block).
- Quote eval JSON `ffn_moe.mean_cost_*` (deployed argmax), not training-time `mean_cost` (one microbatch, exploration noise). A collapsed router looks identical on CE alone — memory:ce-vs-chance-floor.
- Straight-through coefficient must be `1 + p - p.detach()`, never `p / p.detach()` (NaN on near-zero-probability rungs).
- AC recompute must replay each layer's routing decision (argmax flips at 27B aborted training; fixed 52c32d82c).
- MoE timing: quote `gemm_ms` (balanced bmm floor); `torch._grouped_mm` backward raises and at 27B killed the CUDA context — `--grouped` stays off.
- Library code was moved/docstrings repointed to `research/` scripts in the clean port (3c4e02983, 2026-10-12) — paths above are the landmark-branch layout.

## Verified vs unverified
- Verified 2026-10-12: every Qwen3-4B f1 / eval_size / SE / deployed cost above against the family LAUNCH_LEDGER.tsv RESULT rows; ffnmoe-t10 4B/9B multipliers against scale_report.md; git history of `nested_ffn_moe.py` and the family (last landmark commit 2026-09-05; clean-port commits 2026-10-12).
- Read only in memory (not re-opened): H100 speed numbers, the 2026-09-21 MoE-vs-null table (spot-checked: top-8 best gemm_ms 6.54 at c=1/16 matches `ffn_moe_arm_4B_32768.json`; dense 26.88 / null 1.35 / top-2 1.90 not re-derived), 2026-09-09 layer-pruning findings.
- Memory `nested-ffn-moe-task` describes this as an ongoing study ("first 4B wave scored"); later evidence (fs35 grid 09-02→04, flexcompute 09-09, MoE pricing 09-21, ffndrop 09-29) shows the line closed here and moved on. Marked `done`. UNVERIFIED — confirm with Prasann that no further standalone ffnmoe runs are planned.
