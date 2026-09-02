# ffnmoe — nested-width FFN mixture (flexible-compute FFN, learned router)

**Question: how little FFN compute can a token get away with?** A learned per-token router picks
one of several *nested* FFN widths — full, 1/4, 1/16, 1/64, or a zero-cost null rung — and a
budget hinge loss pushes the mean per-token FFN cost down while cross-entropy holds it up.

This family is a **standalone study of the FFN axis**: plain causal attention, no KV compaction,
no soft tokens, no pooled-doc-KV. Composing it with the pooled-doc-KV compression
(`../pooledkv/`) is a later step and deliberately not wired here — if both axes move at once,
neither result is interpretable.

Mechanism, and how it relates to AdaMoE / MatFormer / MoNE: `olmo_core/nn/nested_ffn_moe.py`.

## Why nested widths and not AdaMoE

AdaMoE adds zero-FLOP **null experts** to an existing top-k MoE, so the number of *real* experts
per token varies. Two things make it a poor fit here: it presupposes an MoE (Qwen3-4B is dense,
one MLP per layer), and all its experts are the same size, so the cheapest nonzero option is still
a full expert — hence its ~14% FLOP saving, not 10–100×. The nested-width geometry
(MatFormer / MoNE) gives a token a *little* MLP instead of forcing full-or-nothing, and the null
rung is kept as the bottom of the ladder. Nesting adds **no expert parameters**, so this can be
fitted onto an already-trained checkpoint.

## Why this should not repeat the role-gated failure

The role-gated FFN (`enable_role_gated_ffn`) gave every context-doc token the same binary
full-or-nothing choice from a deterministic rule. Gate@4 could not fit the training data at all
(CE wall 0.95); gate@12 fit but evaluated 0.316. See `records/pooled-doc-kv-attention.md`.

Three differences here:

1. **A ladder, not a switch.** A token that needs a little MLP takes 1/64 instead of choosing
   between full and zero.
2. **Zero-init router.** It selects the full rung with p ≈ 1, so step 0 is *bit-identical* to the
   base model and any degradation is attributable to compute actually given up.
3. **Hinge on the batch mean, not a per-token pull to zero.** The model may spend a full FFN on a
   few tokens as long as the average is under target — which is the entire point of adaptivity.
   Below target the term is exactly 0 and only CE is optimized.

## Status

First 4B wave done (2026-09-01): contradiction n=100, Qwen3-4B, 3 epochs, plain causal, held-out
`contradiction_eval_pubmed_both_n100_k3` (eval_size=488, binomial SE ±0.005–0.011). The
`mean_cost` column is the hard routing on the LAST training microbatch of the routed layers
(exploration noise included -- see the caveat below); "total FFN" folds in the unrouted early
layers; "total FLOPs" also folds in attention (at 6k context) and the LM head.

| arm | routed layers | target | mean_cost (routed) | total FFN FLOPs | total model FLOPs | f1 | EM |
|---|---|---|---|---|---|---|---|
| dense | – | – | 1.000 | 1.0× | 1.0× | **0.988** ±0.005 | 0.965 |
| v1 `t05-L4` | 4–35 | 0.05 | 0.064 (15.7×) | 6.1× | 1.8× | 0.937 ±0.011 | 0.816 |
| v2 `t15-L4` | 4–35 | 0.15 | 0.159 (6.3×) | 4.0× | 1.7× | 0.975 ±0.007 | 0.926 |
| v3 `t05-L12` | 12–35 | 0.05 | 0.065 (15.4×) | 2.7× | 1.5× | 0.976 ±0.007 | 0.928 |

Read: **the model can drop ~94% of its FFN compute on the late layers for ~1 SE of f1** (v3), and
routing the early layers (4–11) is where the accuracy goes (v1 vs v3: same cost per routed
layer, −0.04 f1). The router uses the ladder as full / 1⁄16 / 1⁄64 / null; the 1⁄4 rung is
essentially never chosen. Rung usage at the end of v3: full 5.8%, 1⁄4 1.7%, 1⁄16 2.0%, 1⁄64 7.2%,
null 83%. The *full* rung is 90% of the residual cost, so going further is about routing fewer
tokens to full, not adding cheaper rungs.

**Caveat on this wave: the anneal schedules were not resume-safe.** The holder's schedule clock
was in-memory only, so each sneetches SIGSEGV resume restarted the target/exploration anneals
from 1.0 / 0.1. v1 resumed three times and ended at `target=0.95 explore=0.09`; v3 ended at
`target=0.84 explore=0.08`; only v2 was near-settled. The routers had already found the cheap
regime before the resets (the hinge is 0 below target, so a reset target does not push cost back
up), which is why the cost numbers still hold, but the three arms did NOT get the same schedule
and v1's extra loss may partly be its extra noise. Fixed (the callback pins the clock to
`(global_step − 1) × calls_per_step` in `pre_step`). Second wave, launched 2026-09-01 22:40 with
no activation checkpointing and the fused ladder kernel: `v5-t05-L12-clean` (clean repeat of
v3), `v4-t01-L12-d1024` (the 100× arm; ladder `1,16,64,1024` -- the baked base fixes the rung
count at 4 divisors + null) and `dense-noac` (throughput baseline + second dense sample). Plus `v6-t05-L0` (launched 23:05): **every layer routed** -- the start layer was a
prior from the role-gate failure, not a constraint of the method, and the ladder + batch-mean
budget let the router keep early layers at full width by itself if CE needs it.

**Quote deployed costs from a batch-1 eval only.** The batch-8 eval left-pads prompts to the longest in the group, pad tokens are routed too, and the router (which never saw pads) sends most of them to the full rung: v4 read 0.060 padded vs **0.0077 (130×) at batch 1**, with identical f1. `BATCH_SIZE=1 OUT_TAG=-bs1` on the eval launcher.

The eval script now also reports the **deployed** cost -- hard argmax routing over the entire eval
(prefill + decode), no exploration -- as `ffn_moe.mean_cost_routed_layers` /
`mean_cost_all_layers` in the result JSON. Quote that next to f1, not the training-time number.

### Routing ALL layers (second wave, no AC, fixed schedule, batch-1 deployed costs)

Prasann asked for the all-layers version and, when it fell short, to iterate. Deployed costs are
hard argmax routing over the whole eval at batch 1 (no pads). eval_size=488, SE ±0.010–0.013.

| arm | change | deployed FFN cost (all 36 layers) | f1 | EM | where the router spends |
|---|---|---|---|---|---|
| dense (two runs) | – | 1.0 | 0.988 / 0.982 | 0.965 / 0.947 | – |
| v5 `L12/0.05` (reference) | only L12–35 routed | 0.37 (17× on the 24 routed layers) | **0.976** | 0.926 | L12 0.74, L15 0.36, rest ≤0.07 |
| v6 `L0/0.05` | all layers | 0.051 (19.5×) | 0.916 | 0.752 | L0 1.00, L7 0.52, L3 0.20, rest ~0 |
| v7 `L0/0.05 + curriculum` | routing opens L35→L0 over the first half | 0.060 (16.7×) | 0.906 | 0.719 | L0 1.00, L3 0.96, L6 0.15; ladder collapsed to full/null |
| v8 `L0/0.10` | 2× the budget | 0.109 (9.2×) | 0.943 | 0.834 | L0 1.00, L5 1.00, L7 0.80, L14 0.76 |
| v9 `L0/0.05, two-stage` | warm-start from v5, open L0–11 with the hinge active from step 0 | 0.050 (20×) | **0.939** | 0.822 | L0 1.00, L8/L9/L19 ~0.20 |
| v11 `L0/0.05, two-stage + fine ladder` | v9's recipe with the 7-rung ladder, warm-start from v10 | 0.042 (24×) | 0.938 | 0.824 | L0 1.00, L10 0.15, L5 0.11, L6 0.09 |
| v12 `L0/0.02, two-stage + fine ladder` | same at a 0.02 budget | 0.025 (40×) | 0.929 | 0.787 | L0 0.85, everything else ~0 |

Read:

1. **The router protects the early layers by itself.** In every all-layer arm it runs layer 0
   fully dense and picks two or three other early layers to keep; the "start at L4/L12" prior
   was half right (a few early FFNs matter) and half wrong (not the first twelve).
2. **The accuracy loss is the budget, not the mechanism.** At 0.05 on all layers (2.1× total
   model FLOPs) the best arm is 0.04 below dense; at L12+ the same per-layer budget costs
   ~0.01. Doubling the budget (v8) or spending it smarter (v9) recovers half the gap.
3. **Curriculum hurts**: opening layers late collapses the ladder to full/null (v7).
4. **Two-stage is the best all-layer recipe at a fixed budget** (+0.023 over v6 at the same
   cost), and it converges fast -- it was under target by step ~1100.
5. **The fine ladder does not close the all-layer gap.** Combining the two winners (v11) gives
   the same f1 as v9 at a lower cost (24× vs 20×); pushing the budget to 0.02 (v12) costs
   another 0.01. On all layers the accuracy is set by the budget, and the ladder only changes
   how cheaply a given budget is met. The all-layer f1 ceiling at ≤0.05 sits at ~0.94 in every
   variant tried, ~0.04 below dense -- the early layers (1–11) want more than a 20× budget.

What is left to try for a full-model 20× at dense accuracy: a longer/gentler anneal (the
two-stage arm reached target in a third of the run, so there is schedule slack), the local
reconstruction loss (`--ffn-moe-recon-weight`, still never turned on), or a per-layer floor
that exempts L0 from the budget mean (it sits at 1.0 anyway, so exempting it frees 1/36 of the
budget for the layers that actually vary).

### The 100× ladder and the single-unit rung (L12+, target 0.01)

| arm | ladder (widths) | deployed cost on L12–35 | f1 |
|---|---|---|---|
| v4 | `1,16,64,1024` → 9728/608/152/8 | 0.0077 (**130×**) | 0.963 |
| v10 | `1,16,64,256,1024,9728` → 9728/608/152/38/9/1 | 0.0079 (**127×**) | **0.973** |

With the 4-rung ladder, 130× costs 0.013 f1 vs the 17× arm (v5, 0.976). **With intermediate
rungs down to a single hidden unit (v10, Prasann's suggestion) the same 127× costs nothing
measurable: 0.973 vs 0.976, within 1 SE.** Deployed, v10 puts 25% of tokens on the width-1 rung,
9% on 1/256, 63% on null and 0.7% on full: the model wants a *tiny* MLP for a quarter of its
tokens, and a ladder that jumps from 1/1216 to nothing forced those onto rungs that were either
too expensive or too small. Bake a 7-rung base (`WIDTH_MULTIPLE=1 DIVISORS=1,16,64,256,1024,9728`)
to use it; the rung count is fixed by the base.

### Does the FLOP saving turn into speed? (measured)

Only a little at this shape, and after fixing the kernel the reason is the FFN's share of the
step, not the routed path. Three measurements, all in `debug/ffnmoe/` (scripts + logs):

**1. One FFN layer, isolated (`profile_routed_layer.py`, A100, 6144 tokens, v3 rung mix, 15×
fewer FLOPs).** The first routed implementation was CPU-bound: 0.85 ms of GPU kernels but 3.2 ms
of wall, because each rung's `nonzero` drained the GPU queue (five host syncs per layer) and the
narrow kernels then paid their launch cost serially. Its backward was worse for a different
reason: autograd's handling of the weight *slices* -- `SliceBackward` allocates a full-size zero
gradient per rung and `add_`s four full 50 MB weight gradients per matrix -- was 2× the GEMM
time. Both are fixed by `_NestedLadderFn`, one autograd node for the whole ladder that writes
each weight gradient once and needs one host sync per layer:

| one FFN layer, A100 | dense | routed, first version | routed, fused ladder |
|---|---|---|---|
| forward | 6.70 ms | 3.17 ms (2.1×) | **1.51 ms (4.4×)**, GPU-bound |
| forward + backward | 14.9 ms | 11.0 ms | 11.5 ms (GPU time 4.0 → 3.0 ms; wall is CPU-bound) |

The remaining backward gap is eager-mode CPU overhead of ~60 small kernels; closing it needs a
compiled or Triton backward, not more FLOP cuts.

**2. Whole training step, single H200 (`profile_train_step.py`, 1×6144 tokens, bf16, fused
loss, v3 mix routed from L12).** Activation checkpointing was ON in the first wave's launcher
(`--ac-mode full`), which recomputes every block in backward; the default is now `none`.

| full step, H200 | AC full | no AC |
|---|---|---|
| dense | 741 ms | 603 ms |
| routed (fused ladder) | 759 ms | 573 ms (1.05×) |
| floor: routed layers' FFN deleted outright | 650 ms (1.14×) | 497 ms (1.21×) |

So even a *free* FFN on L12–35 buys 1.21× here. Of the dense 603 ms, the forward alone shows
attention 82 ms, FFN 46 ms and the fused-linear LM head (151k vocab) 140 ms; the rest is
backward and eager elementwise work (casts, RMSNorm, residual adds -- as much GPU time as all
the GEMMs in the uncompiled model). Per token at 6k the FFN is 55% of *FLOPs* but ~18% of
*wall-clock* in this setup. A compiled model would raise the FFN's share (and the routed path's
narrow kernels are compile-hostile until shapes are static -- see below).

**3. Decode (8 sequences × 1 token per forward).** Routed is 0.65× dense (63 vs 41 ms): at 8
tokens every FFN is launch-bound and routing only adds launches. Per-token routing cannot speed
decode; skipping the FFN *weight read* at decode needs batch-level routing (all 8 null) or a
batch of 1.

**What would convert more of it.** (a) Static-capacity ("expert-choice") routing: each rung takes
its top-C tokens by router score, shapes are fixed, zero host syncs, compile/CUDA-graph friendly,
and the budget is enforced exactly by the capacities instead of a hinge -- a different training
objective (MoNE-style), so a new arm. (b) `torch.compile` on the model for the non-FFN 80%.
(c) Longer contexts move time *toward* attention, so the FFN axis matters less there, which is
the case for composing with the KV axis later rather than pushing this one past ~1.2×.

### Can it go to 100×?

By configuration, yes: `resolve_rung_widths` takes any divisor list (widths floored to a multiple
of 8, minimum 8), so `DIVISORS=1,16,64,1024` gives widths `[9728, 608, 152, 8]` + null (the base is baked with a 5-rung router, so a ladder must keep 4 divisors -- a 5-divisor ladder fails at load with a `[5]` vs `[6]` size mismatch on `_nffn_gain`) and
`TARGET=0.01` asks for 100× on the routed layers. That is the `v4-t01-L12-d1024` arm. What it
tests is not the ladder but whether the router can cut *full*-rung traffic from ~6% of tokens to
under 1% without the accuracy following -- at v3's usage the full rung alone already costs 0.058.
On the whole model, 100× on L12+ is worth 1.56× total FLOPs versus 1.52× at 15× (see above) and,
per the step measurements, no wall-clock beyond the 1.21× floor; the interesting number from v4
is the f1, not the speed.

### Bugs the smoke test and first GPU launches caught (all fixed, all now regression-tested)

1. **NaN CE by step 75 (CPU smoke).** The straight-through router coefficient was `p / p.detach()`,
   whose gradient is `1/p` — and exploration deliberately routes tokens to ~0-probability rungs.
   The additive form `1 + p - p.detach()` has the same forward value and constant gradient.
   Guarded by `test_straight_through_gradient_is_bounded_for_improbable_rungs`.
2. **`CheckpointError` at step 0 (first 4B launch).** Activation checkpointing re-runs each block
   in backward; the exploration draw came from the ambient RNG, so the recompute put a different
   number of tokens on each rung (5653 vs 5643) and torch aborted. Routing draws are now seeded
   per `(seed, call index, layer)` — `holder.calls` advances only in `begin_forward`, which the
   recompute doesn't re-enter. Guarded by `test_routing_is_deterministic_within_a_forward`.
3. **Silent monitor for ~200 steps × 2 runs.** Three separate causes, all fixed: the callback
   gated its own recording on an interval offset from the trainer's collect interval; it cached
   the holder in `post_attach`, but the instance that receives `post_step` is a different one, so
   the cache was always `None`; and `metrics()` returned `{}` when the live accumulators had been
   reset, making an inert router indistinguishable from a broken monitor. The holder is now
   resolved fresh each step, `metrics()` falls back to a snapshot of the last completed forward
   and always reports schedule state, and the callback writes a plain `[ffn-moe]` console line in
   addition to `record_metric`. **The console line is the reliable one** — read it, not the
   metric block.

## Files

| file | what |
|---|---|
| `bake_ffn_moe_into_base.py` | writes a base copy WITH the router/gain keys (+ optional importance permutation). **Run this first** — the routed arm cannot load a base without them. |
| `Qwen3-4B-ffnmoe-contra-local.py` | the trainer. `--ffn-moe-start-layer -1` is the dense reference arm. |
| `run_q4b_ffnmoe.sbatch` | launcher (`FFN_MOE_START`, `TARGET`, `DIVISORS`, `BUDGET_W`). |
| `eval_q4b_ffnmoe_contra.sbatch` | held-out contradiction eval; refuses to score a run whose recorded routing differs from the requested flags. Result JSON carries the deployed `ffn_moe` cost. |
| `LAUNCH_LEDGER.tsv` | every job id, arm, status, result. |
| `debug/ffnmoe/bench_ffn_routing.py`, `profile_train_step.py` | (repo `debug/`) the layer-level and whole-step speed measurements quoted above. |

## Setup (once, on the training node)

The bake step is required, and the importance permutation is free accuracy — nested rungs use the
FIRST `width` hidden units, so without reordering a 1/64 rung is an *arbitrary* 1/64 of the MLP.
Permuting hidden units (`w1`/`w3` rows and `w2` columns together) is an exactly output-preserving
reparameterization, so sorting them by importance costs nothing.

```bash
mkdir -p /data/prasann/ffnmoe_exp
cp -r /data/prasann/pooledkv_exp/contra_n100_qboth_train /data/prasann/ffnmoe_exp/
python src/scripts/train/memexpress/ffnmoe/bake_ffn_moe_into_base.py \
  --base /data/prasann/pooledkv_exp/q4b-dense-cpt-fixmark/model_and_optim \
  --out  /data/prasann/ffnmoe_exp/q4b-dense-cpt-fixmark-ffnmoe \
  --permute weight --start-layer 4 --divisors 1,4,16,64
```

## Running

```bash
cd src/scripts/train/memexpress/ffnmoe
# dense reference (no router) -- the number every routed arm is compared against
sbatch -w sneetches -p jsteinhardt -q preemptive_high --gres=gpu:H200:2 \
  --export=ALL,RUN=q4b-ffnmoe-dense,FFN_MOE_START=-1 run_q4b_ffnmoe.sbatch
# routed arm, asking for a 20x mean FFN reduction from layer 4 on
sbatch -w sneetches -p jsteinhardt -q preemptive_high --gres=gpu:H200:2 \
  --export=ALL,RUN=q4b-ffnmoe-v1,FFN_MOE_START=4,TARGET=0.05 run_q4b_ffnmoe.sbatch
```

Then score, with the **same** routing flags the run trained with:

```bash
RUNS="q4b-ffnmoe-v1" FFN_MOE_START=4 sbatch -w sneetches -p jsteinhardt \
  -q preemptive_high --gres=gpu:H200:2 --export=ALL,RUNS,FFN_MOE_START \
  eval_q4b_ffnmoe_contra.sbatch
```

## Reading the run

`NestedFFNMoECallback` logs the HARD (executed) routing, not router probabilities:

- `ffn_moe/mean_cost` — mean per-token FFN cost on routed layers, as a fraction of dense.
- `ffn_moe/speedup` — `1 / mean_cost`, the FFN-only speedup on routed layers.
- `ffn_moe/frac_rungK` — fraction of tokens on rung K (0 = full, last = null).
- `ffn_moe/target`, `ffn_moe/explore` — the annealed schedules.

**Watch `mean_cost` and CE together.** The two failure modes are a router that never leaves the
full rung (`mean_cost` pinned at 1.0 — raise `BUDGET_W`) and one that dumps everything on null
(`frac_rung<last>` → 1.0 with CE climbing — lower `TARGET`, raise `--ffn-moe-recon-weight`, or
push `--ffn-moe-start-layer` later). Without this callback the CE curve looks the same either way,
which is exactly how the role-gate arms burned runs.

## Knobs worth sweeping first

| knob | default | why |
|---|---|---|
| `TARGET` | 0.05 | the whole result — how much compute you are demanding back |
| `FFN_MOE_START` | 4 | both role-gate failures were about early layers |
| `--ffn-moe-recon-weight` | 0.0 | local full-FFN distillation on 2% of tokens; the most direct form of "same output, smaller FFN". FFN-only, so no attention and no full-context forward is involved |
| `--router-lr` | 1e-3 | the router starts cold while the backbone is pretrained |
| `--ffn-moe-width-multiple` | 8 | rung widths are floored to a multiple of this. `1` allows a **single-hidden-unit** rung: `DIVISORS=1,16,64,9728` → widths `[9728, 608, 152, 1]` (v10). Recorded in the checkpoint's `ffn_moe` block; the eval reads it from there |
| `--ffn-moe-layer-curriculum-frac` | 0.0 | opens routing from the LAST layer down to `FFN_MOE_START` over this fraction of training (v7). Closed layers run dense and sit outside the budget mean. The per-layer cost line (`[ffn-moe] step N per-layer cost:` every 100 steps, and `ffn_moe.per_layer_mean_cost` in the eval JSON) shows where an all-layer router spends |
| `DIVISORS` | `1,4,16,64` | `1,16,64,1024` is the 100× ladder (v4); the 1/4 rung goes unused, so drop it. Keep exactly 4 divisors: the baked base fixes the rung count |
