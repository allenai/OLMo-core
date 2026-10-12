# Can the routers reach a 40x training speedup?

Working directory for the flexible-compute speedup question: FFN-width routing, KV-keep routing
and block skipping, priced against the model's own FLOP formulas and then measured on hardware.

## Files

| file | what it does |
|---|---|
| `ceiling.py` | Decomposes per-token training FLOPs into FFN / GDN / attention projections / attention scores / LM head at four model configs and seven context lengths, then prices every router operating point. Writes `ceiling.json`. |
| `bench_head.py` | Measures the LM-head gating win on a GPU: the head alone (fwd+bwd, both loss implementations) and a full model step. |
| `bench_head.sbatch` | Runs the above on one H200. |
| `gdn_bottleneck.py` | Interleaved, drift-controlled batch/heads sweep of the delta-rule kernel with a repeated control, plus a CPU-enqueue-vs-wall split and per-kernel attribution. Resolves the contradictory sequential sweeps. |
| `gdn_levers.py` | A/B of the candidate GDN speedups in one process: CUDA graph, `cu_seqlens` packing, gc on/off; plus a full 4B GDN block stock vs graphed, and a measured device-copy bandwidth for an honest roofline. |
| `step_launch_bound.py` | Full-model fwd+bwd: wall / GPU-busy / idle / launch count, eager and `apply_compile()`, so the isolated launch-bound result can be checked against a real step. |
| `gdn_chunksize.py` | Forward-only fla chunk-size sweep (C = 16/32/64/128), each arm CUDA-graph captured. Tests chunk size as a traffic lever. |
| `gdn_expandv.py` | `linear_value_head_dim` 128 vs 64 at three levels (isolated kernel eager + CUDA-graphed, 4B GDN block, full 4B step), interleaved with a control. Carries the corrected byte model (`traffic_bytes()`) that includes `h`/`dh`. |
| `ffn_moe_arm.py` | Prices a **standard top-k MoE** on the same axis as the nested-ladder arms: E experts holding `E*W_e = F_hidden`, swept over k/E, plus the real shipped Qwen3.5 MoE geometries (35B-A3B / 122B-A10B / 397B-A17B) dropped into each dense scale's `d_model`. Reports the balanced-capacity bmm (== `MoEMLP.forward`, a floor) and the full gate+permute+combine path. `--grouped` opts into `torch._grouped_mm`, which is off because its backward raises and at 27B killed the CUDA context. Writes `ffn_moe_arm_<scale>_<T>.json`. |
| `make_null_moe_panels.py` | Regenerates the "Null rungs, narrow rungs, and a standard MoE" section of `research/kv_route/viz/ffn_routing_linearity.html` from the two sweeps' JSON. |
| `mlskip_router_overhead.py` | Prices ONE block-skip routing decision: each prologue component in isolation, then a real 4B attention block and GDN block split into `stock` / `+key-mask only` / `+router+mix only` / `full`, then a real K-block stack routed per-block vs per-span. Reports launches per arm and checks span exactness. |
| `mlskip_static_prune.py` | The degenerate case of multi-layer skipping: an always-null FFN measured as (a) dense, (b) the deployed nested-FFN path pinned to the null rung, (c) structurally deleted -- at FFN level and on a full Qwen3-4B step, with the always-null layer set read out of `rungs_v12-*.json` rather than assumed. |
| `mlskip_span_step.py` | Span skipping end-to-end: full 4B fwd+bwd with all 32 blocks routed, routers PINNED so every arm skips the same tokens, span in {1,2,4,8} with and without the lazy end-of-span mix, at matched keep. Wall / GPU-busy / idle / launches. |

## The finding that reframes the question

Setting FFN, GDN, projections **and** attention scores to literally zero caps Qwen3.5-4B at
**5.8x at 2k, 6.4x at 8k, 9.0x at 32k**. No combination of the three routers can exceed that,
because none of them touches the LM head — 17.3% of per-token training FLOPs at 2k, 11.2% at 32k
(the "embeddings + LM head" bucket in the cost-anatomy artifact is *entirely* the head; embeddings
are a lookup and contribute no FLOPs).

The head is also almost pure waste on a masked-label fine-tune. Measured loss density on real
shards:

| shard | loss density | head compute discarded |
|---|---|---|
| `pooledkv_exp/contra32k_train` | 0.00175 | 99.83% |
| `pooledkv_exp/contra_n100_qboth_train` | 0.00642 | 99.36% |
| `xlong5_qafter/shards_chunked/rerank_train` | 0.05104 | 94.90% |

Gating the head to supervised positions lifts the three-router floor from 7.7x to **50x at 32k**
(40.2x at 2k, 42.5x at 8k). It is the term that decides whether 40x is reachable at all.

## Caveats that survive the arithmetic

- **Quality is the real wall.** The "3router floor" row is a mechanism limit, not a trained point:
  the measured three-router floor held oolong at 0.583 (dense 0.667) and put contradiction at 0.
  Head gating is the exception — it is exactly loss-neutral, since the positions it drops
  contribute nothing to the loss today.
- **FLOPs are not wall-clock.** The routed FFN is 4.4x faster in forward but moved step time only
  1.05x. Every number here is FLOPs unless it came from `bench_head.py`.
- **Head gating applies to masked-label SFT only.** Under full-sequence LM pretraining the loss
  density is 1.0 and there is nothing to gate.

## Measured (H200, sneetches)

Head in isolation, fwd+bwd, 4B geometry. Loss is bit-identical in every row (`loss_rel_err` 0.0).

| ctx | density | unfused CE | unfused peak | fused CE (our training path) |
|---|---|---|---|---|
| 2k | 0.0064 | 9.9x | 6.97 -> 2.45 GB | 4.18x |
| 8k | 0.0064 | 37.6x | 24.1 -> 2.50 GB | 1.12x |
| 32k | 0.0018 | 151x | 92.8 -> 2.62 GB | 1.36x |
| 64k | 0.0018 | dense OOM | OOM -> 2.80 GB | 2.17x |

**The memory win is on the unfused path only.** Our trainer already uses fused-linear CE, which
never materialises the logits, so at the model level gating saves ~0.4 GB of 62 -- call it zero.
What gating buys under fused CE is time, not memory. The unfused column matters only as the reason
the fused workaround exists.

### Full-model step speedup, after the unfused fallback

| model | ctx | fused only | with unfused fallback |
|---|---|---|---|
| 0.8B | 2k | 2.61x | **3.33x** |
| 0.8B | 8k | 1.52x | **2.70x** |
| 0.8B | 32k | 1.16x | **1.35x** |
| 4B | 2k | 2.05x | **3.03x** |
| 4B | 8k | 1.03x | **1.60x** |

Exactness, attributed on one model (`check_kernel_delta.py`): dense vs gated with both sides on the
SAME implementation gives rel 0.00e+00 (default AND fused), so the gather is bit-identical. Dense
fused vs dense unfused gives 1.26e-03, and the end-to-end gated-vs-dense difference is also
1.26e-03 -- i.e. the entire residual is the deliberate kernel swap, none of it the gather. The
unfused side is plain `cross_entropy` over materialised logits, so the swap moves toward the
reference implementation rather than away from it.

**The fused kernel has a large N-independent cost.** Gated time barely tracks the number of
surviving rows (13 rows still costs 70 ms), because the Liger backward touches a full
`d_model x 248320` fp32 gradient accumulator every step regardless. That is a floor gating cannot
remove, and it is why the isolated head does not collapse the way FLOPs predict.

## Implementation

`LMHeadConfig.sparse_loss_positions` (default off). When the caller wants a reduced loss and no
logits, `LMHead.forward` gathers hidden states to the positions where `labels != ignore_index`
before the final norm and the vocabulary projection. Both are position-wise, so this is exact
rather than an approximation — verified in `src/test/nn/lm_head_sparse_loss_test.py`, which checks
the loss *and* the input gradients match the dense path. Declines to fire under `loss_reduction
="none"`, when logits are requested, and under tensor/context parallelism. Exposed on the trainer
as `--sparse-loss-positions`.


## FFN wall-clock (2026-09-08)

### Does routed-FFN wall-clock scale linearly with routed-FFN FLOPs?

`bench_ffn_linearity.py` sweeps the cost fraction c = mean width / full width from 1 to 1/1024 at
4B FFN geometry and fits `time = a + b*c`. **Yes -- almost exactly -- but only once the step has
enough tokens to make the GEMMs compute-bound.**

| tokens | curve | a (ms) | b (ms) | R2 | floor as % of full | max FFN speedup |
|---|---|---|---|---|---|---|
| 2048 | dense-narrow | 1.029 | 0.255 | **0.27** | 71.4% | 1.4x |
| 2048 | ladder-1rung | 1.335 | 0.344 | **0.28** | 70.7% | 1.4x |
| 2048 | ladder-mixed | 2.518 | **-0.683** | 0.58 | 132.9% | **0.8x (slower)** |
| 8192 | dense-narrow | 0.583 | 4.601 | 0.988 | 11.3% | 8.9x |
| 8192 | ladder-1rung | 0.698 | 5.787 | 0.986 | 10.5% | 9.5x |
| 8192 | ladder-mixed | 1.311 | 5.374 | 0.986 | 19.3% | 5.2x |
| 32768 | dense-narrow | 0.809 | 21.860 | **0.998** | 3.6% | 28.1x |
| 32768 | ladder-1rung | 1.930 | 24.335 | 0.993 | 7.3% | 13.8x |
| 32768 | ladder-mixed | 2.094 | 25.352 | **0.9985** | 7.5% | 13.3x |

At 2k tokens there is effectively NO relationship (R2 0.27, slope ~0): FFN time is width-independent,
the GEMMs are latency/bandwidth bound, and full routing is *slower* than dense. At 32k the fit is
near-perfect and the routing floor is only 7.5%, so the FFN axis converts FLOPs to time at ~13x.
The routing machinery roughly doubles the floor over an ideally-narrow FFN (7.5% vs 3.6%), which is
what separates 13.3x from 28.1x.

### But the FFN is a small share of the step

`bench_ffn_share.py` ablates one component at a time from a real fwd+bwd step (H200):

| model | tokens | step | FFN share | mixer share | head-gate share | FFN ceiling |
|---|---|---|---|---|---|---|
| 4B | 8k | 0.791 s | 22.1% | 35.4% | 36.0% | 1.28x |
| 4B | 16k | 1.429 s | 26.2% | 48.0% | 22.3% | 1.35x |
| 0.8B | 2k | 0.334 s | 2.0% | 23.3% | 70.2% | 1.02x |
| 0.8B | 8k | 0.377 s | 5.2% | 24.7% | 63.3% | 1.05x |
| 0.8B | 32k | 1.016 s | 9.4% | 58.2% | 26.0% | 1.10x |
| 0.8B | 64k | 2.509 s | 7.7% | 74.7% | 12.7% | 1.08x |

Shares are not additive (ablating one changes what the others contend for) and the FFN ablation is
generous -- it removes the FFN's parameters from the backward too.

**Conclusion: the routed-FFN kernel is not the problem.** It converts FLOPs to time essentially
linearly at realistic token counts. The problem is that the FFN is 49-55% of 4B FLOPs but only
22-26% of wall-clock, so even a perfect FFN router caps the STEP at 1.28-1.35x.

### Splitting the mixer: it is GDN, not attention (`bench_mixer_split.py`)

Lumping attention and GDN into one "mixer" number averages away the effect. Ablated separately,
priced against each family's own FLOPs (H200 bf16 peak ~990 TF/s):

| model | tokens | step | ATTN ms | wall% | TF/s | GDN ms | wall% | TF/s |
|---|---|---|---|---|---|---|---|---|
| 4B | 8k | 0.785 s | 106.0 | 13.5% | **385** | 150.8 | 19.2% | **115** |
| 4B | 16k | 1.409 s | 330.4 | 23.4% | **407** | 296.7 | 21.1% | **117** |
| 0.8B | 8k | 0.378 s | 34.8 | 9.2% | **346** | 54.9 | 14.5% | **62** |
| 0.8B | 32k | 1.011 s | 387.6 | 38.3% | **431** | 192.1 | 19.0% | **71** |
| 0.8B | 64k | 2.500 s | 1475.3 | 59.0% | **441** | 385.6 | 15.4% | **71** |

Achieved throughput, all three families: **FFN ~595 TF/s (60% of peak), attention 385-441 (~40%),
GDN 62-117 (~12%)**.

* **Full attention is MORE efficient than the model average** (ratio 0.79 at 4B/16k: 23.4% of
  wall-clock for 29.6% of FLOPs). It is large at long context because the quadratic term is
  genuinely large, not because it is slow. Not worth optimising on efficiency grounds.
* **GDN is the disproportion**: ~7.7-8.7% of FLOPs but 19-21% of wall-clock, ratio ~2.4x.
* **Why**: training dispatches GDN to `chunk_gated_delta_rule` (`recurrent.py:366`; the
  `fused_recurrent` variant is decode-only). That is chunk-parallel within a chunk but a
  SEQUENTIAL scan across chunks, plus a short conv, gates, per-chunk states and an fp32 output
  norm -- all low arithmetic intensity. A GEMM streams its weights once and does O(N) work per
  byte; the delta rule runs a scan over small matrices. GDN is structurally incapable of GEMM-like
  throughput, which is what 115 vs 407 TF/s measures.

**So the best wall-clock target in the model is GDN, not the FFN** -- 2.4x over-represented, and
GDN block-skipping is already built and GPU-tested (`src/test/nn/block_skip_gdn_test.py`).

Measurement caveat for all of the above: batch=1 single sequence, which matches our real
micro-batch (`--micro-batch-instances 1`) but gives GDN's scan less parallel work to hide behind
than a wider batch would. Timings sync only at step boundaries, so intra-step overlap is preserved.


## GDN q/k/v fusion: implemented, exact, and NOT worth it (2026-09-08)

`GatedDeltaNet(fuse_qkv=True)` replaces the three q/k/v projections with one 2560->12352 GEMM and
the three depthwise convs with one over 8192 concatenated channels. A state-dict pre-hook
concatenates `w_q`/`w_k`/`w_v` and the three conv weights on load, so every existing split
checkpoint loads unchanged. (Upstream HF Qwen3.5 already stores these FUSED --
`nn/hf/convert.py:490,504` splits them on import -- so the fused layout is the natural one.)

**Exact**: output bit-identical (`max |dout| = 0.000e+00`). The input gradient differs by
`max |dgrad| = 7.812e-03` against a gradient max of `9.570e-01` -- **rel 8.16e-03, i.e. 1.05x bf16
epsilon**: a single ULP, which is rounding from the fused dgrad accumulating in a different order
than three separate ones, not a backward bug. (Judge it against the gradient MAX; the 4.75e-02
"rel-to-rms" figure compares a max-error to an rms-magnitude and is not like-for-like.)

**Speed: 1.019x at block level -- inside the noise.**

| version | split | fused | ratio |
|---|---|---|---|
| first attempt (split-then-cat before the conv) | 9.581 ms | 10.516 ms | **0.911x** |
| no-copy, run 1 | 10.032 ms | 9.845 ms | **1.019x** |
| no-copy, run 2 | 10.278 ms | 9.856 ms | **1.043x** |

In BOTH post-fix runs the p10s overlap and split's best case is faster than fused's (run 1: split
p10 9.528 vs fused 9.730; run 2: 9.585 vs 9.670). The medians move because split's tail is longer
(p90 12.140 / 13.130 vs 10.267 / 11.000), so the 1.02-1.04x is split's variance, not fused's speed.

**Why the isolated sizing (1.19x projections, 3.66x convs) did not translate.** Fusion removes
kernel launches and improves GEMM shape, but it does not reduce HBM traffic, which is what binds
here. The fused projection reads `x` once instead of three times (~84 MB saved, ~28 us at 3 TB/s)
against a ~10 ms block; the weights are read once either way and the 134 MB qkv output is written
either way. GDN at 115 TF/s vs the FFN's 595 is bandwidth/kernel-bound in a way fusing does not
address.

**Verdict: leave `fuse_qkv` off.** It is exact and harmless, and it is the correct layout, but it
buys nothing measurable. Speeding up GDN needs less traffic (e.g. a smaller `expand_v`) or fewer
GDN token-passes (block skipping), not better-shaped kernels.


## Why GDN is slow: RESOLVED (2026-09-08) -- SUPERSEDED

> **Read the next section instead.** Its verdict (occupancy-bound) and all three of its levers
> were refuted by interleaved, drift-controlled measurement later the same day. Kept because the
> refutation of hypothesis A (FLOP accounting) still stands and because the reasoning is the
> counter-example that motivated the re-measurement.

GDN is 7.7-8.7% of counted FLOPs but 15-21% of step wall-clock (~115 TF/s vs the FFN's ~595).
Three candidate causes, measured by `resolve_gdn_gap.py` rather than inferred:

| hypothesis | verdict |
|---|---|
| **A. `num_flops_per_token` undercounts the chunked kernel** | **REFUTED** -- analytic chunked count is 4,194,304 FLOP/token, exactly **1.00x** the formula's `2*4*state_size`. The four intra-chunk `C*D` terms (4 x 64x128) sum to the same 32,768 MACs as the two `DK*DV` state terms at this geometry. The FLOP shares and the routing arithmetic built on them are safe. |
| **B. bandwidth-bound** | **REFUTED** -- 15.8% of HBM peak at best. |
| **C. occupancy / latency-bound** | **CONFIRMED** |

`chunk_gated_delta_rule` alone, fwd+bwd, 4B geometry (H200: ~990 TF/s bf16, ~4.8 TB/s HBM):

|      T |     ms | ms/token |  TF/s | %peak |  GB/s | %peak |
|---|---|---|---|---|---|---|
|   2048 |  1.426 | 0.6965us |   9.0 |  0.9% | 282.3 |  5.9% |
|   8192 |  2.362 | 0.2883us |  21.8 |  2.2% | 681.9 | 14.2% |
|  32768 |  8.499 | 0.2594us |  24.3 |  2.5% | 758.0 | 15.8% |

**Neither roofline binds** (2.5% compute, 15.8% bandwidth), and per-token cost falls 0.697 -> 0.288
-> 0.259 us and then PLATEAUS. That is a scan with a fixed dependency chain: at 32k there are
T/C = 512 sequential chunk steps, each with only 32 heads x 128x128 of state to occupy the GPU.

**This explains the fusion null result correctly.** Fusion removes launches and improves GEMM shape;
the binding constraint is a dependency chain, which reshaping does not touch. 1.02x was the right
answer for a reason not identified until now. (An earlier claim in this file that GDN is
"bandwidth-bound" was wrong -- it was inferred from the negative fusion result, not measured.)

**Levers that actually address a scan-length constraint**, in order of cost:
1. **Larger chunk size** -- C=64 today; doubling it halves the sequential steps and doubles
   intra-chunk parallel work. A config knob, not a kernel rewrite. Try first.
2. More batch/head parallelism to fill the machine during the scan.
3. Fewer GDN token-passes (block skipping -- already built and GPU-tested).

NOT a lever: reducing traffic (e.g. a smaller `expand_v`). Bandwidth is not the limit.

## Why GDN is slow: RE-RESOLVED, and the previous answer was wrong (2026-09-08, later)

The section above concluded "occupancy/latency-bound" by ELIMINATION and named larger chunk size,
more batch/head parallelism and fewer GDN passes as the levers. Three of those four claims are now
refuted by direct measurement. New scripts: `gdn_bottleneck.py`, `gdn_levers.py`,
`step_launch_bound.py`, `gdn_chunksize.py` (all with their `.sbatch`; all run on cubbins, H200).

### First: the contradictory sweeps were drift, and the drift is CPU-side

`gdn_occupancy_test.py` ran its batch and heads sweeps sequentially and measured the same config
2.364 ms in one and 3.734 ms in the other. `gdn_bottleneck.py` re-runs both INTERLEAVED in one
process with the control point (B=1, H=32, T=8192) re-measured inside every round. The control's
round medians:

| round | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| ms | 2.841 | 2.379 | 2.377 | 3.537 | 4.417 | 4.213 | 4.375 | 4.647 |

**1.955x spread on an unchanged configuration**, monotonically rising to a plateau. That is larger
than any effect either sweep claimed to find, and it fully accounts for the 58% discrepancy: the
batch sweep ran while the process was still in its fast regime and the heads sweep after it had
settled into the slow one. `gdn_levers.py` alternated `gc.enable()`/`gc.disable()` by round and the
drift was identical (3.037 / 3.325 / 4.572 / 4.818 / 4.743 / 4.770 ms), so it is not Python GC. It
is CPU-side: the CUDA-graph arm of the same work has p10/median/p90 = 2.134 / 2.144 / 2.172 ms,
**p90/p10 = 1.018x, and no drift at all**.

*Discipline note for anything else measured in this directory: eager timings on this box carry a
~2x within-process drift. Capture the arm in a CUDA graph and time replays if you want the GPU
number.*

### The interleaved sweeps, and why both were still misleading

| axis | work | wall scaling | "free" | GPU-busy scaling (profiler) |
|---|---|---|---|---|
| batch B=1->8 (H=32) | 8x | 3.74x | 53% | 2.125 -> 16.270 ms = **7.66x** |
| heads H=32->64 (B=1) | 2x | 1.15x -> 1.21x | 85% (H=8->64) | 2.125 -> 3.767 ms = **1.77x** |

The wall-clock "free" fractions are artifacts. Measured GPU-busy time scales **~linearly with work
on both axes** (96% of ideal on batch, 89% on heads). There is no idle capacity to fill.
**Occupancy is REFUTED**, and with it "more batch/head parallelism" as a lever.

### What actually binds, part 1: CPU launch cost (in isolation only)

At B=1, H=32, T=8192, fwd+bwd of `chunk_gated_delta_rule` alone:

- GPU-busy **2.125 ms** against a **4.418 ms** clean wall -> **51.9% of the wall is GPU idle**.
- Time for the CPU to enqueue one iteration with no sync: **4.667 ms**, i.e. **99.9%** of the
  pipelined wall. The CPU is the slower side, not merely blocked by the GPU.
- 25 kernel launches per iteration. Profiler CPU attribution puts the cost inside fla's own Python
  dispatch: `ChunkGatedDeltaRuleFunctionBackward` 33.5%, `ChunkGatedDeltaRuleFunction` 25.3%, with
  `aten::*` and `cuLaunchKernelEx` together under 12%.

Capturing the same work in a CUDA graph (`gdn_levers.py`):

| arm | p10 | median | p90 | vs control | CPU ms | CPU% |
|---|---|---|---|---|---|---|
| control (B=1,H=32) | 2.979 | 4.635 | 4.889 | 1.000x | 4.644 | 99.9% |
| base (same config, own tensors) | 3.169 | 4.630 | 4.974 | 1.001x | 4.599 | 99.9% |
| varlen4 (T as 4 packed docs) | 2.992 | 5.047 | 6.285 | **0.918x** | 4.942 | 99.9% |
| varlen8 | 3.002 | 5.105 | 6.114 | **0.908x** | 4.906 | 99.9% |
| varlen32 | 3.006 | 5.041 | 6.368 | **0.919x** | 4.948 | 99.9% |
| **CUDA graph** | 2.134 | 2.144 | 2.172 | **2.162x** | 0.013 | 0.6% |

Full 4B GDN block (projections + convs + delta rule + gated norm), same treatment:

| arm | p10 | median | p90 | CPU ms | CPU% |
|---|---|---|---|---|---|
| block | 7.951 | 11.708 | 12.586 | 13.389 | 99.8% |
| block, CUDA-graphed | 6.204 | 6.212 | 6.234 | 0.082 | 1.3% |

**1.885x at block level**, and the graphed median lands on the GPU-busy time, which is the
independent confirmation that the missing 5.5 ms was launch cost and nothing else.

**Packing is not a lever** (0.91-0.92x, slightly slower — `cu_seqlens` adds CPU-side chunk-index
prep and does not change the launch count), even though it multiplies the scan grid. That is one
more independent refutation of the scan-parallelism story.

### What actually binds, part 2: memory bandwidth (the GPU time itself)

The earlier section refuted bandwidth at "15.8% of HBM peak". That figure came from a traffic
model counting only q/k/v/o. The chunked algorithm also materialises the per-chunk state
`h` = (B, T/C, H, K, V) — **128 MiB** at this shape, the single largest tensor in the block — plus
`dh` at the same size, and neither was in that model. Redone:

- **MEASURED** achievable device copy rate on this H200: 128 MiB copy in 75.7 us = **3.55 TB/s**
  (74% of the 4.8 TB/s spec). All percentages below are against 3.55, not the spec sheet.
- **CALCULATED** traffic per kernel, from the tensors each fla launcher actually passes.
- Kernel times are **MEASURED** (profiler, per iteration, B=1 H=32 T=8192).

| kernel | calls | ms | calc. traffic | achieved | % of 3.55 TB/s |
|---|---|---|---|---|---|
| `recompute_w_u_fwd_kernel` | 2 | 0.168 | 2 x 290 MiB | 3.61 TB/s | 102% * |
| `chunk_fwd_kernel_o` | 1 | 0.134 | 385 MiB | 3.01 TB/s | 85% |
| `chunk_bwd_kernel_dqkwg` | 1 | 0.327 | 898 MiB | 2.88 TB/s | 81% |
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 2 | 0.334 | 2 x 385 MiB | 2.42 TB/s | 68% |
| `prepare_wy_repr_bwd_kernel` | 1 | 0.270 | ~514 MiB | 2.00 TB/s | 56% |
| `chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64` | 1 | 0.309 | 513 MiB | 1.74 TB/s | 49% |
| `merge_16x16_to_64x64_inverse_kernel` (solve_tril) | 1 | 0.160 | 96 MiB | 0.63 TB/s | 18% |

\* over 100% means the byte count is slightly generous — the small `A` tiles get L2 reuse. Read it
as "at the copy rate", not as a bandwidth violation.

Aggregated over all 25 kernels the calculated traffic is **~5.1 GB per iteration** against the
**measured 2.125 ms** of GPU-busy time = **2.41 TB/s = 68% of achievable**. The same arithmetic on
the compute side: fwd+bwd is ~1.03e14 FLOP (3x the analytic chunked forward count), giving
**48.5 TF/s = 4.9% of the 990 TF/s peak**.

Those two numbers are one statement. Arithmetic intensity is **20 FLOP/byte** against this
machine's balance point of 990/3.55 = **279 FLOP/byte**, so the chunked delta rule can never exceed
**7.2% of compute peak** on an H200 no matter how it is scheduled — and it already achieves 68% of
that ceiling. **GDN is memory-bound. Hypothesis B was right and was wrongly dismissed.**
(Incidental: `resolve_gdn_gap.py` prices fwd+bwd as `3/2 x` the forward count where it should be
`3 x`, so its TF/s column reads 2x low. It does not change any conclusion there.)

### At model level, most of the launch cost is already hidden

`step_launch_bound.py`, full fwd+bwd of the real model, batch 1:

| model | ctx | compile | wall ms | GPU-busy ms | **idle** | launches | fla+conv kernels |
|---|---|---|---|---|---|---|---|
| 4B | 8192 | - | 559.3 | 546.1 | **2.4%** | 6104 | 65.5 ms (12.0% of GPU-busy) |
| 4B | 8192 | yes | 503.0 | 498.7 | **0.9%** | 3358 | 68.6 ms (13.8%) |
| 4B | 16384 | - | 1223.7 | 1219.5 | **0.3%** | 6141 | 128.5 ms (10.5%) |
| 0.8B | 8192 | - | 161.6 | 150.7 | **6.8%** | 4219 | 31.6 ms (20.9%) |
| 0.8B | 8192 | yes | 157.9 | 128.6 | **18.6%** | 2159 | 32.1 ms (25.0%) |

(The `cpu ms` column that script also prints is **not** a valid launch-bound indicator at this
scale: with ~6000 launches per step the ~1024-deep CUDA queue fills and the CPU blocks inside
`cudaLaunchKernel`, so `cpu ~ wall` regardless. `idle` is the trustworthy column here; the CPU
numbers are only trustworthy for the 25-launch isolated sequences above.)

So **CUDA-graphing GDN would NOT deliver 1.9x on a 4B step today** — there is only 0.3-2.4% of idle
to recover, because neighbouring layers' GPU work covers the GDN blocks' launch cost. Two things
follow anyway:

1. At **0.8B with compile the step is already launch-bound** (18.6% idle, 2159 launches). Graphing
   is worth ~1.2x there.
2. The margin at 4B is thin. The isolated GDN mixer costs **13.4 ms of CPU per block**; 24 of them
   is 321 ms of a 559 ms eager step. Any change that removes GPU work without removing launches
   walks into that floor — which is the mechanism behind the older observation in this file that a
   **routed FFN 4.4x faster in forward moved step time only 1.05x**. Routers must cut launches, not
   just FLOPs.

Also worth separating: the ablation-based "GDN = 19.2% of wall" mixes the delta rule with GDN's
seven projections and with launch cost the ablation also removes. Directly attributed, the
**non-GEMM GDN kernels (delta rule + conv + l2norm + solve_tril) are 12.0% of GPU-busy at 4B/8k for
~0.4% of model FLOPs** — a ~30x disproportion, and it is a bandwidth disproportion.

### Larger chunk size: not reachable, and measured not worth it

The previous section's lever #1 ("a config knob, not a kernel rewrite. Try first") is wrong on both
counts. `fla.ops.utils.solve_tril` ships only 16x16 / 32x32 / 64x64 merge kernels and asserts
`A.shape[-1] in [16, 32, 64]`, so C=128 raises `AssertionError` — confirmed empirically. Reaching
it means authoring a 128x128 triangular-inverse kernel, plus the two backward kernels
(`chunk_gated_delta_rule_bwd_dhu`, `prepare_wy_repr_bwd`) that hardcode `BT = 64` and ignore their
`chunk_size` argument.

`gdn_chunksize.py` sweeps the reachable direction, forward only, each arm CUDA-graph captured
(p90/p10 <= 1.004, so these differences are real):

| C | median us | `h` tensor | vs C=64 |
|---|---|---|---|
| 16 | 874.9 | 512 MiB | 0.769x |
| 32 | 693.6 | 256 MiB | 0.970x |
| 64 | 672.7 | 128 MiB | 1.000x |
| 128 | `AssertionError` in `solve_tril` | 64 MiB | — |

Outputs agree to 7.69e-03 relative (bf16 rounding), so the arms compute the same thing.
**Doubling C from 32 to 64 halves the `h` tensor and buys 3.1%.** The curve has already flattened,
because the traffic C removes from `h` it adds to the intra-chunk `A` (T x H x C fp32) and the C x C
matmuls. Extrapolating, C=128 is worth ~1-2% — not worth writing three kernels for. **Lever
refuted.**

### Summary of levers

| lever | verdict |
|---|---|
| More batch/head parallelism | **REFUTED** — GPU-busy already scales linearly with work on both axes. |
| Document packing (`cu_seqlens`) | **REFUTED** — 0.91-0.92x, slightly slower. |
| Larger chunk size | **REFUTED** — unreachable past 64, and the measured slope says ~1-2% if reached. |
| q/k/v + conv fusion | **REFUTED** earlier (1.02x); now explained: it changes neither traffic nor launch count materially. |
| CUDA graphs / fewer launches | **REAL but situational** — 2.16x on the kernel, 1.885x on the block, ~1.0x on a 4B step (0.3-2.4% idle), ~1.2x on a compiled 0.8B step (18.6% idle). |
| Less traffic | **The only lever with real headroom**, and the previous section explicitly ruled it out on a bad byte count. |

### What to try next, in order

1. **`expand_v`.** `h` is (B, T/C, H, K, **V**) and `v`/`u`/`v_new`/`dv`/`do`/`o` are all ~V-sized.
   Dropping `expand_v` 2.0 -> 1.0 (value_dim 4096 -> 2048, DV 128 -> 64) cuts roughly 40% of the
   traffic, so up to ~1.5x on the GDN kernels — 12% of GPU-busy at 4B/8k, so ~1.05x step. Needs a
   quality check; it is a model change, not a kernel change. The previous section listed this under
   "NOT a lever" on the strength of the bad bandwidth number.
2. **A GVA-native kernel.** `recurrent.py:411-414` `repeat_interleave`s q and k from 16 to 32 heads
   before the call, doubling their bytes in the ~8 kernels that read them. fla's kernels index q/k
   per v-head, so this needs kernel support for `n_v_heads > n_heads`, not a call-site change.
3. **Graph the mixer where idle actually exists** — 0.8B, and any routed/block-skipped config that
   cuts GPU work. Not worth it for dense 4B today.
4. **Not worth doing**: saving `h` from forward instead of recomputing it in backward saves 0.167 of
   2.125 ms (8%) and costs 128 MiB of activation per layer (3 GB at 4B/8k). Bad trade.

### Follow-up: both recommended levers, measured (2026-09-08, later still)

New script: `gdn_expandv.py` / `.sbatch`. Same discipline: interleaved A/B in one process, repeated
control, p10/median/p90, nothing compared across jobs.

**Naming correction first.** The earlier note called this "`expand_v` 2.0 -> 1.0". Wrong for this
model: the 4B reaches value_dim 4096 via `linear_num_value_heads=32` x `linear_value_head_dim=128`,
and its `expand_v` is **already 1.0** (`expand_v = linear_value_head_dim / linear_key_head_dim`).
The real knob is `linear_value_head_dim` 128 -> 64, i.e. expand_v 1.0 -> 0.5, DV 128 -> 64,
value_dim 4096 -> 2048.

#### Priority 1: halving the value head dim

CALCULATED traffic with the corrected byte model (h and dh included), T=8192, H=32, DK=128, C=64 —
the model is now encoded in `traffic_bytes()` in the script rather than living in prose:

| | DV=128 | DV=64 |
|---|---|---|
| total per fwd+bwd | 4658.5 MiB | 3602.5 MiB |
| largest term | `chunk_bwd_dqkwg` 834.0 | `chunk_bwd_dqkwg` 610.0 |

Ratio 0.773, so a perfectly bandwidth-bound kernel should go **1.293x** faster. Measured:

| level | arm | p10 | median | p90 | p90/p10 | vs DV=128 |
|---|---|---|---|---|---|---|
| kernel, **eager** | DV128 | 3.737 | 4.570 | 4.724 | 1.264x | 1.000x |
| kernel, **eager** | DV64 | 3.693 | 4.682 | 4.857 | 1.315x | **0.976x** |
| kernel, **CUDA-graphed** | DV128 | 2.129 | 2.137 | 2.146 | 1.008x | 1.000x |
| kernel, **CUDA-graphed** | DV64 | 1.892 | 1.895 | 1.903 | 1.006x | **1.128x** |
| 4B GDN block, eager | DV128 | 6.674 | 13.236 | 14.176 | 2.124x | 1.000x |
| 4B GDN block, eager | DV64 | 6.412 | 13.963 | 14.187 | 2.213x | 0.948x (noise) |
| **full 4B step** | DV128 | 543.754 | 546.240 | 552.172 | 1.015x | 1.000x |
| **full 4B step** | DV64 | 506.871 | 507.762 | 512.637 | 1.011x | **1.076x** |

Control drift in the same process: eager kernel 1.230x, graphed kernel **1.003x**, eager block
**2.050x**, step 1.005x. The eager block A/B is therefore uninformative — both arms are CPU-bound
and the drift is bigger than any effect — and is reported only so nobody re-runs it expecting a
number. The eager *kernel* row is informative in a different way: **DV makes no difference at all
in eager mode (0.976x, inside the spread)**, exactly as the CPU-launch-bound diagnosis predicts,
since halving DV changes bytes but not the 25 launches.

**The traffic cut does NOT convert as predicted: 1.128x measured against 1.293x predicted, i.e.
only 44% of it.** Achieved bandwidth on the graphed (pure GPU) times, against the 3.62 TB/s copy
rate measured in the same process:

| | time | calc. traffic | achieved | % of achievable |
|---|---|---|---|---|
| DV=128 | 2.137 ms | 4658.5 MiB | 2.29 TB/s | 63.2% |
| DV=64 | 1.895 ms | 3602.5 MiB | 1.99 TB/s | **55.1%** |

The kernel gets *less* bandwidth-efficient at DV=64 (narrower V tiles, less work amortising each
byte), which eats more than half the win. The delta rule is memory-bound, but not so cleanly that
traffic and time are interchangeable.

**And most of the step win is not the traffic win.** This is not an iso-parameter comparison:
w_v, w_g and w_out all halve, so the model goes **4.841B -> 4.464B params (-7.8%)** while the step
goes -7.0%. Decomposing the measured 38.5 ms saving (CALCULATED split, MEASURED total):

- delta-rule kernels: 1.128x on the 65.5 ms of fla+conv GPU-busy = **~7.5 ms**
- removed GEMM work: 377M fewer params x 8192 tokens x 2 x 3 = 1.85e13 FLOP at ~595 TF/s = **~31 ms**
- sum 38.5 ms, against a measured 546.240 - 507.762 = **38.5 ms**.

So ~80% of the 1.076x is "the model is 7.8% smaller", which any parameter cut would buy, and the
actual efficiency win is **~1.4% of the step**.

**Cost side.** `w_v`, `w_g`, `w_out` and `o_norm` all change shape in every GDN block, so **no
existing checkpoint can load** — this is a from-scratch architecture change, not a config flip on a
trained model. Validating it needs a full CPT/pretrain run plus the standard eval suite, because
halving the recurrent state's value width is a capacity change to the mixer, not a kernel swap.
**Verdict: not worth that risk for ~1.4%.**

#### Priority 2: the GVA-native path — fla cannot do it, stopping as instructed

`recurrent.py:411-414` `repeat_interleave`s q and k from `n_heads=16` to `n_v_heads=32` before the
kernel call, doubling their bytes for zero FLOPs. By the byte model above, q/k-attributable traffic
is ~1728 of the 4658.5 MiB (**37%**), so a grouped-head kernel would cut ~18% of traffic — worth
about 1.2x on the kernel if it converted fully, and by the DV result above it would not.

It is not available:

- The public `chunk_gated_delta_rule` documents q, k and v as `[B, T, H, K/V]` with a **single** H
  and takes no head-group argument.
- Every file in the call chain (`common/chunk_delta_h.py`, `common/chunk_o.py`,
  `common/chunk_scaled_dot_kkt.py`, `gated_delta_rule/wy_fast.py`, `utils/solve_tril.py`,
  `utils/cumsum.py`, `modules/l2norm.py`) has **zero** occurrences of the grouped-head plumbing
  (`HQ`, `NG`, `num_kv_heads`) that fla does carry in `ops/attn`, `ops/gsa`, `ops/rwkv6`. The only
  matches for that grep are `USE_G`, the gate flag.
- Passing a stride-0 expanded view instead of a materialised copy does not help either:
  `ChunkGatedDeltaRuleFunction.forward` is wrapped in fla's `@input_guard`
  (`fla/utils.py:137-166`), which calls `.contiguous()` on **every** tensor argument, so fla would
  materialise it internally regardless.

Per instruction, not writing a kernel. Recorded as blocked-upstream.

#### Incidental: host syncs in the GDN path (re: the FFN router's `bincount().tolist()`)

Checked for the same pattern. Two exist and neither is a per-layer cost:

- `recurrent.py:314` `bool(cache_leftpad.any())` is a genuine host sync but is short-circuited by
  `cache_leftpad is not None` and only fires on the cached-decode path, never in training.
- `fla/ops/utils/index.py:118` `prepare_chunk_indices` does `.tolist()` on a CUDA tensor and builds
  `torch.cat([torch.arange(n) for n in ...])` in a Python loop over the document count. It fires
  only when `cu_seqlens` is passed (document packing) and is reached from **12** call sites in this
  chain, but it is wrapped in `@tensor_cache`, a single-entry cache keyed by object identity — all
  24 GDN layers get the same `cu_doc_lens` object within a step, so it stays warm and costs one
  sync per step, not per layer. It is still the most likely cause of the packing arms measuring
  0.91-0.92x (slower) earlier in this section.

#### Where this leaves the GDN line of work

The remaining prize is now bounded by measurement, not by argument. Deleting the delta-rule, conv
and l2norm kernels **entirely** (GPU-busy shares from `step_launch_bound.py`, against the measured
walls) would give:

| model | ctx | ceiling if GDN's non-GEMM kernels cost zero |
|---|---|---|
| 4B | 8192 | 1.132x |
| 4B | 16384 | 1.117x |
| 0.8B | 8192 | 1.243x |

The best lever found gets ~1.4% of a 4B step. **Stop optimising the GDN kernels.** What is left,
in order: (1) fewer GDN token-passes — block skipping, already built and GPU-tested, which attacks
the 1.13x ceiling itself rather than the efficiency inside it; (2) the CPU launch floor, which is
already binding at 0.8B+compile (18.6% idle) and is what a routed/skipped config will hit as soon
as it removes GPU work; (3) nothing in the fla kernels.

## Multi-layer skipping: span routing and static layer pruning (2026-09-08)

The block-skip router (`olmo_core/nn/block_skip.py`) decides run/skip **per token, per block**. This
section asks whether a decision should instead cover a SPAN of K consecutive blocks, on the theory
that one decision per K blocks amortises the per-block prologue -- which is CPU/launch work, and
launches are what bind a step that is 559 ms wall against 546 ms GPU-busy. Three scripts, all
interleaved-in-one-process with a repeated control (`mlskip_router_overhead.py`,
`mlskip_static_prune.py`, `mlskip_span_step.py`; H200, cubbins, batch 1).

**Headline: the amortisation is real and measurable, but it is amortising the wrong thing, and the
degenerate case -- deleting a layer outright -- is worth an order of magnitude more.**

### 1. What one per-block routing decision costs (MEASURED, 4B @ 8k, fwd+bwd)

One block, four arms that separate the two things block skipping adds:

| block | stock | `+key-mask only` | `+router+mix only` | full `block_skip` |
|---|---|---|---|---|
| GDN (L0) | 13.960 ms / 180 launches | 13.988 (**+0.20%**) / 194 | 14.951 (**+7.10%**) / 234 | 15.111 (+8.24%) / 248 |
| attention (L3) | 20.907 ms / 200 launches | 37.055 (**+77.24%**) / 272 | 21.467 (**+2.68%**) / 254 | 38.042 (+81.95%) / 326 |

Control spread in the same process 1.032x, arm p90/p10 1.011-1.049x, so every one of these is
outside the noise. At 32k the split is starker: key-mask **+124.07%** on the attention block,
router+mix **+0.24%**; on GDN +4.08% and +7.42%.

**The router prologue costs +54 kernel launches and 0.56-0.99 ms per block. The key mask costs
+72 launches and 16.1 ms on an attention block** -- because passing `block_keep` drops the layer off
the flash path onto the KV-route masked path. These are different problems and only the first is a
routing-frequency problem.

Prologue components in isolation (each includes its backward; the `decide` arm re-evaluates the
router, so the rows are not additive):

| component | 8k | 32k | launches |
|---|---|---|---|
| router GEMM (`RouterOnInput`) fwd+bwd | 0.432 ms | 1.408 ms | 19 |
| decision (sigmoid + `>0`) | 0.239 | 0.779 | 12 |
| `holder.accumulate` | 0.063 | 0.067 | 5 |
| ... of which the `keep.sum().item()` HOST SYNC | **0.030** | **0.039** | 2 |
| straight-through residual mix | **0.547** | **1.851** | 20 |

**The host sync is not the problem here.** 0.030 ms is 0.2% of a GDN block, and it is flat in T --
unlike the FFN router's `bincount(...).tolist()`, this one drains a queue that is already short. The
prologue's cost is the two `(B, T, d_model)` elementwise passes: the mix alone is 0.547 ms at 8k and
**scales with tokens** (1.851 ms at 32k, 3.4x for 4x the tokens), i.e. it is memory traffic, not
launch cost.

### 2. What a span of K amortises (MEASURED)

Implemented as `set_block_skip_span(model, K)` (see §4). Two variants: `span` (one decision, still
one residual mix per block) and `span+lazymix` (one decision AND one mix, applied at the end of the
span against the span's input).

A real 4B step, all 32 blocks routed, routers **pinned** so every arm skips exactly the same tokens
in exactly the same blocks -- the only thing that varies is how often the prologue runs:

| arm | launches | added/block | step ms (keep=1.0) | vs no routing | GPU-busy | idle |
|---|---|---|---|---|---|---|
| no routing | 7917 | - | 808.1 | 1.0000x | 792.6 | 1.9% |
| span=1 (per-block) | 9730 | +56.7 | 959.3 | **1.1872x** | 933.3 | 2.7% |
| span=2 | 9269 | +42.3 | 949.6 | 1.1752x | 926.4 | 2.4% |
| span=2 + lazymix | 8821 | +28.3 | 942.0 | 1.1657x | 918.2 | 2.5% |
| span=4 | 9020 | +34.5 | 946.2 | 1.1710x | 922.7 | 2.5% |
| span=4 + lazymix | 8348 | +13.5 | 933.1 | 1.1547x | 911.2 | 2.3% |
| span=8 | 8892 | +30.5 | 942.9 | 1.1669x | 922.2 | 2.2% |
| span=8 + lazymix | 8111 | **+6.1** | **930.1** | **1.1510x** | 908.4 | 2.3% |

(Control spread 1.003x across the run; p90/p10 within each arm <= 1.008x. Same ordering at
keep=0.5: 1.0926x -> 1.0633x.)

**Span=8 with the lazy mix removes 89.3% of the added launches (1813 -> 194) and 19.3% of the
wall-clock overhead (151.2 ms -> 121.9 ms), i.e. 1.031x on the step.** The launch reduction is
nearly total and the wall reduction is not, which is the whole finding: **the step is only 1.9-2.7%
idle, so removing launches is not what buys the time** -- GPU-busy falls in step with wall (933.3 ->
908.4), and what actually goes away is the K-1 skipped residual mixes' memory traffic.

The premise this experiment was built on -- "one routing decision can cover K blocks: K times fewer
router evaluations, gathers and syncs, which is what actually binds" -- is therefore **half right**.
The K-fold reduction happens exactly as predicted (56.7 -> 6.1 added launches per block, and
decisions per step 32 -> 4). It is worth 3%, not because the amortisation failed but because a 4B/8k
step is not launch-bound in the first place; `step_launch_bound.py` said so earlier in this file
(2.4% idle) and this is that number binding again.

Returns are also flat past K=4: span=4+lazymix 1.1547x vs span=8+lazymix 1.1510x. **The remaining
1.15x is the key-mask path**, which no routing frequency touches.

Isolated K-block stacks agree (8k): per-block overhead 33.40% at K=8, span 30.51%, span+lazymix
27.52%; added launches per block 82.6 -> 59.1 -> 34.6. (The K=4/K=8 stacks include attention blocks
L3/L7, which is why their overhead % is far above the K=1/K=2 GDN-only stacks -- a composition
effect, not a span effect.)

### 3. Static pruning of always-null layers: the degenerate case, and the real win

`rungs_v12-t02-L0-d6-warmv10.json` (trained Qwen3-4B FFN router, target .02) has **22 of 36 layers
>99% null**; `rungs_v11` (target .05) has 13. For such a layer the entire per-token apparatus runs
every forward to express a constant. `mlskip_static_prune.py` reads the layer set out of that file
and measures the three states.

One layer's FFN, fwd+bwd:

| arm | 8k | 2k | launches (8k) |
|---|---|---|---|
| dense | 5.708 ms | 1.410 ms | 24 |
| routed, pinned to the **null** rung | **2.038 ms** | **2.058 ms** | 76 |
| routed, pinned to the **full** rung | 7.951 ms | 3.370 ms | 110 |
| structurally absent | 0.006 ms | 0.011 ms | 0 |

**Saying "do nothing" costs 2.03 ms per layer, and that cost does not move between 2k and 8k
tokens** -- it is pure fixed overhead (76 launches: router GEMM, argmax, `bincount().tolist()`,
stable argsort, split, the ladder call and its `zeros_like` output). At 8k it is 36% of a dense FFN;
at 2k it is 145% of one, i.e. *more expensive than simply running the FFN*.

Full Qwen3-4B step (36 layers, 8192 tokens, fwd+bwd, control spread 1.002x):

| configuration | step | vs dense | launches |
|---|---|---|---|
| dense (all 36 FFNs) | 969.3 ms | 1.000x | 6960 |
| **routed**: null on the 22, full rung on the other 14 | 895.7 ms | 1.082x | 9362 |
| **pruned**: those same 22 FFNs deleted, 14 dense | 823.2 ms | **1.177x** | 5864 |
| routed everywhere, all null | 807.6 ms | 1.200x | 8886 |
| all 36 FFNs deleted | 729.4 ms | 1.329x | 5159 |

**The trained router's own operating point is worth 1.082x. The identical function, expressed by
deleting the layers instead of routing them, is worth 1.177x** -- and 1096 fewer launches than the
*dense* model rather than 2402 more. The gap, `routed -> pruned`, is **1.088x = 72.5 ms of a 823 ms
step, spent entirely on expressing constants.**

That gap decomposes (CALCULATED from the per-layer MEASURED values, against the MEASURED total):
22 always-null layers x 2.032 ms + 14 full-rung layers x (7.951 - 5.708) ms = 44.7 + 31.4 =
**76.1 ms predicted vs 72.5 ms measured**. Both halves matter: the null layers pay for machinery
they never use, and the 14 layers that *do* route pay 1.39x over a dense FFN to reach the full rung
through the ladder.

Caveats, in order of importance:

- **This is not iso-quality, but it is close to it by construction.** Deleting an FFN removes the
  router's ability to ever fire it. On these checkpoints it fires for <1% of tokens in those layers,
  so the change touches <1% of token-layer decisions -- but "<1% of decisions" is not "0% of loss",
  and pruning must be validated on the eval suite, not assumed.
- Pruning is a **post-training, deployed-model** change: train with the router, read the per-layer
  null fraction, delete the layers that are ~always null, drop their router. No retraining, no new
  recipe, no kernel.
- The 14 non-null layers still carry the routed path. Their 31.4 ms is a separate problem (the
  ladder's full-rung overhead), not addressed here.

### 4. The span prototype: implemented, exact, off by default

`set_block_skip_span(model, K)` in `olmo_core/nn/block_skip.py`. Existing behaviour is unchanged:
`span` defaults to 1 and nothing calls the regrouping unless asked (`enable_from_config_block` reads
an optional `"span"` key from `config.json`). Conventions kept exactly: hard decision in the
forward, straight-through `1 + p_sel - p_sel.detach()`, keep-all init (bias +10), two-sided annealed
budget -- and the budget still averages over **blocks**, not decisions, so a target of 0.5 means the
same thing at every span.

Mechanically: routed blocks are grouped into runs of K; only the first block of each run keeps a
router, the rest read the decision back out of the holder. The dropped routers leave the state dict
and the parameter list (stashed on the holder, which is not an `nn.Module`, so `set_block_skip_span(model, 1)`
restores them). `holder.accumulate` takes the whole span's layer list, so the `.item()` sync and the
depth bincount are paid once per span while `per_layer_keep` / the FLOP meter / `BlockSkipCallback`
still see every block -- no callback changes needed.

`lazy_span_mix` (on by default when span > 1) defers the `where`/straight-through to the last block
of the span and applies it once against the span's input. **This is exact, not an approximation**: in
every block of the span a skipped token is excluded as a key (attention) and does not write the
recurrent state (GDN), so no kept token's output depends on what a skipped token's hidden state does
inside the span, and the skipped token's own value is replaced by the span input at the end.

Exactness, MEASURED on the 4B stack at keep-all init, and pinned by
`src/test/nn/block_skip_span_test.py` (6 CPU tests):

- `span` vs `per-block`: **max |d| = 0.000e+00, bit-identical** at K=4 and K=8.
- routed vs unrouted: 3.46-5.19 bf16 eps (K=4/K=8) -- the residual re-composition `h + coef*(out-h)`
  rounds once per block in bf16. Present at span=1 too; not a span effect.
- `span+lazymix` vs unrouted: **2.31-4.32 bf16 eps, i.e. CLOSER to the unrouted reference than the
  eager mix**, because it rounds once per span instead of once per block. In fp32 (the CPU test) it
  agrees with the eager mix to 1e-6.

**What it costs in expressiveness.** A token that skips a span's head skips every block in it and
cannot re-enter mid-span; the reachable set of per-token depth profiles shrinks from `2^n_routed` to
`2^(n_routed/K)`, and the router must commit K blocks ahead on the span head's input alone. Nothing
here measures whether that hurts quality -- no span model has been trained.

### 5. Recommendation

**Do the static pruning; do not pursue multi-layer skipping as a speed lever.**

1. **Delete always-null layers.** 1.177x on a 4B/8k step, measured, against 1.082x for the router
   expressing the same function. It needs no new mechanism, no retraining and no kernel -- just a
   bake step that reads the per-layer null fraction and removes the FFN and its router. It is the
   largest and cheapest item found in this file since head gating. Validate on the eval suite first;
   it is a <1%-of-decisions change, not a no-op.
2. **Span skipping is built, exact and cheap to keep, but it is worth ~1.03x.** It does exactly what
   it was supposed to do -- 89.3% of the added launches gone, decisions per step 32 -> 4 -- and that
   converts to 3% because a 4B/8k step is 1.9% idle. Leave `span` available (default 1) for the
   *modelling* question of whether a token should commit K blocks ahead; do not adopt it as an
   optimisation.
3. **The thing to fix is the key mask, not the routing frequency.** Block skipping's cost at 4B/8k
   is 1.187x, of which the router prologue is 0.031x and the rest is the attention key-mask path
   (+77% on an attention block at 8k, +124% at 32k). That is the `compact` path's problem, not a
   span problem.
4. **Early exit (token stops at depth d, no re-entry) is not a better lever.** It is the K -> n_layers
   limit of the span curve, and that curve is already flat: span=4+lazymix 1.1547x vs span=8+lazymix
   1.1510x, so its ceiling is ~1.15x against span=8's 1.151x -- CALCULATED by extrapolation, not
   measured. It removes the last few prologues and touches none of the key-mask cost. Not worth
   building for speed. (It remains interesting as a *modelling* choice; that is a different question.)

Measurement caveats: batch 1, single sequence, eager (no `apply_compile`, no CUDA graphs), so the
launch counts are the eager ones -- `step_launch_bound.py` shows compile roughly halves them, which
would shrink the span prize further. All wall-clock numbers are medians of interleaved arms with the
control re-measured every round; nothing is compared across jobs.

## The scale-resolved WALL-CLOCK ceiling, and the stack that wins (2026-09-09)

How large a TRAINING wall-clock speedup is reachable, across four model sizes and seven context
lengths, in step seconds, with every lever charged its measured cost.

New files: `bench_step_partition.py` (extended to 9B/27B and to activation checkpointing),
`bench_step_partition_scale.sbatch` / `_grid.sbatch` / `_fill.sbatch`,
`ffn_scale_context_sweep.sbatch`, `scale_ceiling.py` (-> `scale_ceiling.json`),
`make_flop_scaling_page.py` (-> `research/kv_route/viz/flop_scaling_ceilings.html`).

### The answer

**5.72x, measured, at 27B/8k.** The recipe is three levers in this order:

> **head gating -> static deletion of always-null FFN layers -> COMPACTED block skip**

| model | ctx | basis | dense | head gate | static FFN | **best stack** | + KV router | plain-equiv | router-only |
|---|---|---|---|---|---|---|---|---|---|
| 4B | 16k | plain | 1417 ms | 1.286x | 1.200x | **5.650x** (10.4x FLOP, conv 0.54) | 5.402x | 5.65x | 4.39x |
| 4B | 16k | **packed x32** | 1121 ms | 1.396x | 1.203x | **3.901x** (conv 0.38) | 3.768x | 3.90x | 2.80x |
| 9B | 8k | AC | 1248 ms | 1.253x | 1.314x | **4.441x** (9.9x FLOP, conv 0.45) | 4.256x | 4.72x | 3.54x |
| 27B | 8k | AC | 3550 ms | 1.096x | 1.604x | **5.720x** (13.1x FLOP, conv 0.44) | 5.499x | 5.89x | 5.22x |

4B/16k reproduced across five separate jobs: 5.590 / 5.607 / 5.630 / 5.650 / 5.693x (median 5.63x).
All arms interleaved with a dense control re-measured each round; drift 1.001-1.016x.

**27B is the best scale.** Both the total and the routers' own return (best stack / head gating
alone) now point the same way: 4.39x / 3.54x / **5.22x** at 4B / 9B / 27B, because the FFN is 46% of
a 27B step against 22% of a 4B one. 9B sits low on both, at 8k on the AC basis.

### Correction: the earlier "decline with scale" was a bug, not a finding

An earlier revision of this section reported 27B as the WORST scale (3.79x). That number came from
a broken measurement: **under activation checkpointing, the model's `CheckpointWrapper` forwards
attribute READS to the wrapped block but not WRITES**, so assigning to `wrapper.feed_forward`
registered a new submodule that nothing read, the inner block kept running its real FFN, and the
static-FFN arm silently measured *nothing* at 9B and 27B. Fixed, 27B's `ffn-static` arm goes
1.000x -> 1.604x and its best stack 3.785x -> 5.720x.

The exclusion was the right call at the time: static deletion cannot lose to routing the same
function, and 27B's routed arm measured 1.531x against the static arm's 1.000x, which is why that
arm was dropped from the coefficient and flagged rather than smoothed. (`AcBlock` in
`bench_step_partition.py` was written to avoid exactly this trap -- see its docstring.)

The head-gating mechanism survives: it saves a roughly *fixed* 0.23-0.36 s (the fused-linear-CE
backward touches a full `d_model x vocab` fp32 accumulator regardless of how many rows survive), so
it is worth 1.29x at 4B and 1.10x at 27B and still runs opposite to the routers. It simply no longer
decides the ordering.

### What made the difference: two implementation fixes, zero FLOPs

| fix | effect |
|---|---|
| block skip becomes a **gather** instead of a mask | `bskip-0.25` 0.99x -> `bskipC-0.25` **2.27x** |
| keep-all flash fast path + split full/partial block-mask tiles | `kv-1.00` (drops nothing) 0.753x -> **0.993x** |

Neither removes a FLOP. Before them, the masked paths cost 1.3-1.4x just to be enabled, and the
whole family looked like a net loss.

### The KV router is REDUNDANT, not merely expensive

Compacted block skip already removes a skipped token as a key, by gathering rather than masking.
Adding the KV router on top re-does that work and charges for it: 5.650 -> 5.402x at 4B,
4.441 -> 4.256x at 9B, 5.720 -> 5.499x at 27B, while *raising* the FLOP cut (10.4 -> 11.7x at 4B).
It is out of the recommended stack. This is the clearest case in the file of the FLOP counter and
the clock disagreeing, and the clock being right.

### Static FFN deletion is a stack member

The trained routers **layer-prune on their own**: `rungs_v11-t05` (target c=0.05, realised 0.042)
sends >99% of tokens to the null rung in **13 of 36 layers**, `rungs_v12-t02` in **22 of 36**; only
L0 stays dense. Deleting them beats *routing* the same function (1.200x vs 1.177x at 4B/16k) because
it carries no per-layer prologue (measured at 15.2% on top of the FFN component), no host sync and
no schedule. Backed out of the `ffn-static` arm it removes **65.6% of the FFN component** at 4B,
64.9% at 9B, 81.3% at 27B.

### Two caveats that change what you would plan

**1. Packing roughly halves the prize.** Same 4B, same 16k, same recipe, sequence packed into 32
documents -- which is how real training runs. The dense step falls 1417 -> 1121 ms because packing
already makes attention cheap, and the recipe falls with it: **5.65x -> 3.90x**, conversion
0.54 -> 0.38. Head gating goes the *other* way (1.286 -> 1.396x): the head is untouched by packing
while everything around it got cheaper. **Plan against the packed number.**

**2. Past ~5x the step is launch-bound.** GPU idle goes **0.9% (dense) -> 12.6% (best arm)**: the
levers removed enough GPU work that the CPU can no longer issue kernels fast enough. CUDA graphs are
worth about **1.14x** from here and are the next lever. The constraint has moved from FLOPs to
launches, and no further FLOP removal pays until it moves back.

### What fits on one H200 (measured, not argued)

| model | plain (no AC) | with activation checkpointing | AC peak |
|---|---|---|---|
| 0.8B | to 64k | to 128k | 4.4-22.2 GiB |
| 4B | **OOM past 16k** | to 128k | 18.1-61.7 GiB |
| 9B | **OOM past 8k** | to 128k | 33.4-90.0 GiB |
| 27B | **OOM at every length** | **to 32k**, OOM at 64k | 100.3-123.9 GiB |

27B fits only with checkpointing (bf16 weights 50.1 GiB + grads, no optimizer state) and only to
32k. It has **no plain-basis cell at all**, so its rows below are extrapolated onto the plain basis
by a measured per-component ratio (0.994 head-waste, 0.788 FFN, 0.774 attention, 0.715 GDN, 0.781
floor -- median over the 13 cells carrying both bases).

### The modelled grid -- a FLOOR, not the result

The three stack coefficients are fitted at the anchors and applied across the grid. The model is
**conservative everywhere, and increasingly so with model size**, because it never credits
compaction with shortening the GDN scan or cutting launches:

| model | arm | measured | model | error |
|---|---|---|---|---|
| 4B | `headgate` | 1.286x | 1.292x | +0.4% |
| | `ffn-static` | 1.200x | 1.200x | +0.0% |
| | `bskipC-0.25` | 2.272x | 2.079x | -8.5% |
| | `staticFFN+bskipK-0.25+headgate` | 5.650x | 4.678x | **-17.2%** |
| 9B | `headgate` | 1.253x | 1.252x | -0.1% |
| | `ffn-static` | 1.314x | 1.319x | +0.3% |
| | `staticFFN+bskipK-0.25+headgate` | 4.441x | 4.276x | **-3.7%** |
| 27B | `headgate` | 1.096x | 1.092x | -0.3% |
| | `ffn-static` | 1.604x | 1.436x | -10.5% |
| | `staticFFN+bskipK-0.25+headgate` | 5.720x | 4.134x | **-27.7%** |

**Quote the measured anchors above; read the grid as a floor.** `basis`: **measured** on hardware
without checkpointing; `AC-converted` measured under checkpointing and converted; `modelled`
neither. `cap` is `1 / floor share` -- embedding lookups, residual adds, the final norm,
`zero_grad`, inter-kernel gaps and the surviving head, which nothing here touches.

| model | ctx | basis | step s | floor | head gate | +static FFN | +block skip | +KV router | cap |
|---|---|---|---|---|---|---|---|---|---|
| 0.8B | 2k | **measured** | 0.32 | 4.0% | 3.42x | 3.58x | **9.28x** | 8.90x | 24.8x |
| 0.8B | 4k | **measured** | 0.33 | 5.2% | 3.31x | 3.49x | **8.49x** | 8.13x | 19.3x |
| 0.8B | 8k | **measured** | 0.38 | 6.8% | 2.67x | 2.96x | **7.40x** | 7.09x | 14.8x |
| 0.8B | 16k | **measured** | 0.55 | 7.7% | 1.82x | 2.02x | **5.99x** | 5.74x | 13.0x |
| 0.8B | 32k | **measured** | 1.02 | 7.5% | 1.36x | 1.48x | **5.39x** | 5.16x | 13.4x |
| 0.8B | 64k | **measured** | 2.49 | 5.8% | 1.15x | 1.22x | **5.65x** | 5.41x | 17.4x |
| 0.8B | 128k | AC-converted | 7.54 | 3.7% | 1.06x | 1.10x | **6.53x** | 6.26x | 27.1x |
| 4B | 2k | **measured** | 0.40 | 6.0% | 2.82x | 3.02x | **7.26x** | 6.96x | 16.7x |
| 4B | 4k | **measured** | 0.52 | 7.4% | 2.09x | 2.70x | **6.27x** | 6.01x | 13.6x |
| 4B | 8k | **measured** | 0.79 | 8.4% | 1.58x | 2.06x | **5.21x** | 5.00x | 11.9x |
| 4B | 16k | **measured** | 1.42 | 8.6% | 1.30x | 1.66x | **4.72x** | 4.52x | 11.6x |
| 4B | 32k | AC-converted | 3.01 | 7.8% | 1.16x | 1.42x | **4.77x** | 4.58x | 12.9x |
| 4B | 64k | AC-converted | 7.57 | 6.0% | 1.08x | 1.25x | **5.26x** | 5.04x | 16.6x |
| 4B | 128k | AC-converted | 22.43 | 4.0% | 1.04x | 1.14x | **6.12x** | 5.86x | 24.8x |
| 9B | 2k | **measured** | 0.41 | 8.1% | 2.06x | 2.68x | **5.89x** | 5.64x | 12.4x |
| 9B | 4k | **measured** | 0.61 | 9.1% | 1.58x | 2.29x | **5.12x** | 4.91x | 11.0x |
| 9B | 8k | **measured** | 1.03 | 9.7% | 1.33x | 1.92x | **4.55x** | 4.36x | 10.3x |
| 9B | 16k | AC-converted | 1.94 | 9.8% | 1.19x | 1.69x | **4.35x** | 4.16x | 10.3x |
| 9B | 32k | AC-converted | 4.11 | 8.9% | 1.11x | 1.51x | **4.44x** | 4.25x | 11.2x |
| 9B | 64k | AC-converted | 9.87 | 7.3% | 1.07x | 1.36x | **4.84x** | 4.64x | 13.6x |
| 9B | 128k | AC-converted | 27.13 | 5.3% | 1.04x | 1.22x | **5.57x** | 5.33x | 18.9x |
| 27B | 2k | AC-converted | 0.87 | 8.7% | 1.40x | 2.03x | **4.68x** | 4.49x | 11.5x |
| 27B | 4k | AC-converted | 1.49 | 8.8% | 1.22x | 1.87x | **4.43x** | 4.25x | 11.4x |
| 27B | 8k | AC-converted | 2.79 | 8.7% | 1.12x | 1.70x | **4.27x** | 4.09x | 11.5x |
| 27B | 16k | AC-converted | 5.66 | 8.2% | 1.07x | 1.58x | **4.29x** | 4.11x | 12.2x |
| 27B | 32k | AC-converted | 12.37 | 7.3% | 1.05x | 1.47x | **4.50x** | 4.31x | 13.6x |
| 27B | 64k | modelled | 29.99 | 6.0% | 1.03x | 1.34x | **4.95x** | 4.75x | 16.7x |
| 27B | 128k | modelled | 81.78 | 4.4% | 1.02x | 1.22x | **5.67x** | 5.43x | 22.9x |

Leave-one-out over the measured partition cells: median step-time error 1.0-2.4% per scale,
interior-only worst 8.2%.

### What is measured and what is not

- **Measured**: every partition cell marked `measured`; three full router matrices
  (`router_matrix_4B_16384_v4.json` plain, `router_matrix_9B_8192_ac_v3.json` and
  `router_matrix_27B_8192_ac_v2.json` AC) plus the packed comparison
  (`router_matrix_4B_16384_v3_packed32.json`); the FFN kernel curves at all four scales and seven
  contexts; the FFN router prologue; the rung histograms; every memory limit above.
- **Extrapolated**: the three stack coefficients come from the anchor cells and are applied at every
  other scale and context. All 27B rows on the plain basis. Rows marked `AC-converted`/`modelled`.
- **Not established at all**: task quality. Head gating is exactly loss-neutral because the
  positions it drops contribute nothing to the loss. Static FFN deletion and block skip are not.

## Every lever at once: corrected FFN numbers, the joint sweep, and what binds at each level (2026-09-09)

New files: `bench_joint_sweep.py` / `.sbatch` (arm matrix, routed step partition, floor
decomposition, graph capture -- four phases in one process), `graph_probe.py` with
`graph_probe.sbatch` / `graph_probe_routed.sbatch` / `graph_probe_nosync.sbatch` (what blocks a
CUDA graph, bisected one case per process). Outputs `joint_sweep_*.json`, `graph_probe_*.json`.

### The bug this re-runs: a third of the FFN was never routed

`--ffn-start-layer` / `--ffn-moe-start-layer` defaulted to **12**. On Qwen3.5-4B (32 blocks) that
left 12 of 32 FFNs dense, so `ffn-fnull1.00` -- nominally "every token takes the null rung" -- cut
the FFN to 0.375 of its own share and total FLOPs only to 0.694 of dense. **Every FFN and stacked
number in `router_matrix_*.json` and in the two published artifacts is understated.** Both defaults
are now 0, and both scripts print the routed-layer list at startup
(`# FFN routed layers (32/32): [...]`), so this cannot recur silently.

MEASURED, dense control re-measured every round; drift **1.004x** at 4B, **1.009x** at 27B.

| arm | 4B/16k plain FLOPx | 4B wall was | **4B wall now** | 27B/8k AC wall was | **27B wall now** |
|---|---|---|---|---|---|
| `ffn-fnull0.00` | 1.65 | 1.101x | **1.180x** | 1.269x | **1.360x** |
| `ffn-fnull0.50` | 1.79 | 1.129x | **1.225x** | 1.344x | **1.467x** |
| `ffn-fnull0.75` | 1.87 | 1.140x | **1.242x** | 1.377x | **1.520x** |
| `ffn-fnull0.90` | 1.92 | — | **1.249x** | — | **1.533x** |
| `ffn-fnull1.00` | 1.96 | 1.177x | **1.314x** | 1.524x | **1.748x** |
| `ffn-static` (deployment only) | 1.96 | 1.200x | **1.362x** | 1.604x | **1.867x** |
| `staticFFN+bskipK-0.25+headgate` | 19.83 | 5.650x | **6.420x** | 5.720x | **6.667x** |

The FFN axis converts at a steady **0.65-0.72** at 4B (0.44-0.57 at 27B) all the way from
`f_null=0` to `f_null=1`: the routed FFN is not what stops converting. `ffn-overhead` (router runs,
every token on the full rung) is **0.946x** at 4B and **0.904x** at 27B -- routing all layers costs
5-10% before it saves anything.

### The joint sweep

Head gating on everywhere, block skip compacted + fixed-capacity, KV router excluded (one arm keeps
it to re-confirm). MEASURED wall speedup vs the same-process dense control:

**4B / 16k plain**, dense 1419.0 ms:

| f_null \ keep | 0.50 | 0.25 | 0.125 | 0.0625 |
|---|---|---|---|---|
| **0.50** | 2.345x | 4.091x | 5.930x | **6.464x** |
| **0.90** | 2.342x | 4.097x | 5.872x | 6.441x |
| **1.00** | 2.337x | 4.043x | 5.833x | 6.293x |

**27B / 8k AC**, dense 3620.3 ms:

| f_null \ keep | 0.50 | 0.25 | 0.125 | 0.0625 |
|---|---|---|---|---|
| **0.50** | 1.820x | 3.116x | 4.548x | **5.104x** |
| **0.90** | 1.818x | 3.097x | 4.486x | 5.034x |
| **1.00** | 1.818x | 3.123x | 4.520x | 5.054x |

and the same points priced and profiled (4B/16k):

| arm | FLOPx | step ms | wall | conv | **GPU busy** | launches |
|---|---|---|---|---|---|---|
| dense | 1.00 | 1419.0 | 1.000x | 1.00 | **99.2%** | 6920 |
| `J-f1.00-k0.5000` | 7.8 | 607.2 | 2.337x | 0.30 | 93.5% | 10961 |
| `J-f1.00-k0.2500` | 19.8 | 351.0 | 4.043x | 0.20 | 85.5% | 10060 |
| `J-f1.00-k0.1250` | 45.4 | 243.3 | 5.833x | 0.13 | 72.2% | 9882 |
| `J-f1.00-k0.0625` | 94.9 | 225.5 | 6.293x | 0.07 | **52.8%** | 9886 |
| `J-f1.00-k0.2500+kv0.25` | 25.3 | 359.5 | 3.948x | 0.16 | 85.3% | 10571 |
| `staticFFN+bskipK-0.25+headgate` | 19.8 | 221.0 | 6.420x | 0.32 | 86.7% | 6954 |
| `staticFFN+bskipK-0.0625+headgate` | 94.9 | 169.1 | **8.390x** | 0.09 | **47.0%** | 6740 |

Four things, all measured:

1. **`f_null` stops mattering once block skip is aggressive.** At keep=0.0625, moving `f_null` from
   0.50 to 1.00 changes the step by 2.7%, in the *wrong* direction. The levers are not independent:
   a skipped token's FFN is already gone, so the FFN router re-cuts a shrinking remainder while
   still paying its full per-layer prologue.
2. **The trainable sweep saturates at ~6.5x** (4B) / **~5.1x** (27B). keep 0.25 -> 0.125 buys
   1.44x; 0.125 -> 0.0625 buys 1.09x for another 2x less work. Conversion falls to **0.07**.
3. **The FFN router costs launches.** Dense issues 6920 kernels at 4B (16709 at 27B); the
   routed-FFN arms issue 9880-14700 (25800-38200 at 27B, up to **2.3x** more) for a fraction of the
   work, while the static-FFN arms sit at or *below* dense. That is the whole gap between the
   6.29x trainable arm and the 8.39x static one.
4. **The KV router is negative again**, third independent confirmation: 4.043x -> 3.948x at 4B and
   3.123x -> 3.046x at 27B, while *raising* the FLOP cut. Keep it out of the stack.

**Packing roughly halves the prize** (4B/16k, `--docs 32`, how real training runs): dense
1120.3 ms, best trainable **4.222x**, best static **5.075x**, GPU busy at the best arm **37.5%**.
Plan against the packed number.

**Caveat on the low keeps.** All routed blocks are forced to skip the *same* tokens, so at
keep=0.0625 the 93.75% skipped tokens do no mixing in any block -- they are carried by the residual
stream from embedding to head. Wall-clock-honest (the compacted path's cost depends on the kept
count, not on which tokens) but not a model anyone would train. Read keep<=0.125 as a mechanism
probe, not an operating point.

### The bottleneck moves: the partition re-measured ON THE ROUTED MODEL

Same cumulative-ablation method as `bench_step_partition.py`, with the routers live at each arm's
operating point. Stages are visited once per round, **forward on even rounds and backward on odd
ones**, each stage's time the median over 4 rounds: a strictly sequential chain is not measurable
here, because the process drifts upward and every small tail component came out NEGATIVE on the
first attempt. Stage-0 drift is 1.003-1.017x on every row below.

A stage was added that the earlier partitions did not have: **delete the block itself**. Replacing
a block's *mixer* and *FFN* leaves the block SHELL running -- two RMSNorms and two residual adds
per block over the whole `(B, T, d_model)` stream, forward and backward, 32 times. That is
per-token GPU work no component ablation ever removed, and it turns out to be essentially the
entire floor (`block_shell` 8.4% of a dense 4B/16k step, with `base` measuring **0.0%**
underneath it).

MEASURED, 4B/16k plain, percentages of that arm's own step (stage 0 has head gating OFF so the
gating saving is visible; rows sum to 100%):

| arm | step ms | head waste | FFN | attn | GDN | routing+shell | **floor ms** | **cap from dense** |
|---|---|---|---|---|---|---|---|---|
| dense | 1418.5 | 22.5% | 26.3% | 21.8% | 20.8% | 8.4% | **122.0** | **11.6x** |
| `J-f0.50-k0.5000` | 919.8 | 34.1% | 24.8% | 11.3% | 16.4% | 13.1% | **123.3** | **11.5x** |
| `J-f1.00-k0.2500` | 663.9 | 48.1% | 18.9% | 6.2% | 14.0% | 12.3% | **85.0** | **16.7x** |
| `J-f1.00-k0.0625` | 538.8 | 59.4% | 9.7% | 1.9% | 17.1% | 11.1% | **64.1** | **22.1x** |
| `staticFFN+bskipK-0.25+headgate` | 540.5 | 59.4% | -0.1% | 7.7% | 17.2% | 15.1% | **85.4** | **16.6x** |
| `staticFFN+bskipK-0.0625+headgate` | 494.4 | 64.3% | 0.8% | 3.3% | 18.8% | 12.0% | **63.3** | **22.4x** |

`floor` here is `step - (head waste + FFN + attn + GDN)`, i.e. everything the routers do not
target. **Routing and the block shell are reported as one bucket on purpose**: the ablation turns
the routers off before it deletes the shell, so it measures the shell at *full* sequence length
and books the difference as negative routing overhead. Their sum is the honest quantity; the split
between them is an artifact. (This is also why the first run of this partition, before the shell
stage existed, showed a suspiciously invariant 119.5 ms "base" in every arm -- that was the
routers-off shell cost, a dense constant, not each arm's floor.)

**As the levers bite, attention collapses (21.8% -> 3.3%) and the FFN collapses (26.3% -> 0.8%),
but GDN barely moves in absolute terms: 295 ms dense -> 93 ms, and then flat from keep=0.25 to
keep=0.0625 (92.9 -> 92.9 ms).** GDN is what block skipping is worst at removing: compaction
shortens the scan but the delta rule is memory-bound at ~20 FLOP/byte and gets *less* efficient on
shorter sequences, so a 4x shorter sequence is not 4x cheaper.

**At the best arm the step is: GDN 53%, floor 37%, attention 9%, FFN 2%.** The bottleneck has moved
from the FFN (dense: 26%) to GDN, with the block shell behind it.

**27B/8k AC reproduces every part of this** on a completely different component mix (the FFN is
46.5% of a dense 27B step against 26.3% at 4B). Its partition ran before the block-shell stage
existed, so its `floor` column is the same `step - (head waste + FFN + attn + GDN)` recomputation:

| arm | step ms | head waste | FFN | attn | GDN | **floor ms** | **cap from dense** |
|---|---|---|---|---|---|---|---|
| dense | 3606.0 | 8.5% | 46.5% | 13.4% | 23.0% | **310.1** | **11.6x** |
| `J-f0.50-k0.5000` | 2303.4 | 13.4% | 47.3% | 8.2% | 18.4% | **292.5** | **12.3x** |
| `J-f1.00-k0.2500` | 1487.8 | 20.7% | 39.8% | 5.4% | 20.5% | **202.3** | **17.8x** |
| `J-f1.00-k0.0625` | 1092.1 | 29.0% | 18.1% | 6.8% | 30.3% | **172.6** | **20.9x** |
| `staticFFN+bskipK-0.25+headgate` | 904.3 | 36.3% | -1.1% | 7.7% | 34.8% | **201.7** | **17.9x** |

Same ceiling at dense (11.6x), same rise as block skip gets aggressive (up to 20.9x), and the same
end state: **GDN goes from 23.0% of a dense step to 30.3-34.8% of a routed one and becomes the
largest single component**, overtaking a 46.5% FFN. Its floor behaves identically too --
MEASURED 8.6-9.1% of dense across 2k-8k, 96-98% GPU-busy, a launch count constant at ~4590
regardless of token count, an 88%-per-token / 12%-fixed split, and the same batch invariance
(B=1/T=4096 floor 168.5 ms vs B=2/T=2048 floor 170.1 ms -- same 4096 tokens, same floor).

### The floor, decomposed

`bench_joint_sweep.py --phases floor` puts every block at Identity and sweeps token count and batch
size with the same alternating-direction rounds. MEASURED at 4B:

| B | T | total tok | dense ms | floor ms | norm | head | embed | zero_grad | rest | floor % of dense | **GPU-busy** | launches |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2048 | 2048 | 449.4 | 37.46 | 0.30 | 1.55 | 0.17 | -0.49 | 35.93 | 8.34% | 15.70 ms (44%) | 1787 |
| 1 | 4096 | 4096 | 535.8 | 40.03 | 0.29 | 0.73 | 1.42 | 0.07 | 37.52 | 7.47% | 32.32 ms (86%) | 1787 |
| 1 | 8192 | 8192 | 789.3 | 66.65 | 0.22 | 1.65 | 0.80 | 0.54 | 63.45 | 8.44% | 61.18 ms (96%) | 1787 |
| 1 | 16384 | 16384 | 1414.6 | 123.11 | 0.27 | 1.70 | 1.02 | 0.61 | 119.52 | 8.70% | 117.52 ms (**98%**) | 1786 |
| **2** | 4096 | 8192 | 762.2 | 66.65 | 0.13 | 1.70 | 0.82 | 0.56 | 63.43 | 8.74% | 61.09 ms (96%) | 1785 |

Least-squares fit `a + b*tokens` over the B=1 rows (MEASURED points, CALCULATED fit): the whole
floor is **17.38 ms fixed + 6.329 us per 1k tokens**; the block-shell remainder is
**15.55 ms + 6.208 us/1k**; the gated head 1.22 ms + 0.030; embeddings 0.61 ms + 0.029; the final
norm 0.28 ms; `zero_grad` under 0.8 ms at 16k.

Three facts settle whether the ceiling moves:

- **The floor is 86% per-token and 14% fixed.** At 16k only 17.4 of 123.1 ms is amortisable --
  **1.2% of the dense step**. A bigger step cannot buy back more than that.
- **Batch does not amortise it.** B=2/T=4096 and B=1/T=8192 are the same 8192 tokens and give the
  same floor to three digits (66.65 vs 66.65 ms). The floor tracks *total tokens*, full stop.
- **It is real GPU work, not launch overhead.** The launch count in the floor configuration is
  **constant at ~1786 regardless of token count** while GPU-busy rises linearly and reaches **98%
  of the floor's wall-clock at 16k**. There is nothing here for CUDA graphs to recover.

**The fused-CE head's fixed cost is NOT hiding in the floor.** MEASURED head cost vs supervised
rows, 4B/16k: **2.12 / 2.10 / 2.28 / 8.67 / 33.05 / 331.54 ms** at 1 / 16 / 128 / 1024 / 4096 /
16384 rows. Flat at ~2.1 ms below ~128 rows, so the N-independent term exists and is **0.15% of the
dense step**. The old worry from the isolated-head work ("13 rows still costs 70 ms") does not
apply at model level, because `LMHead` drops the fused kernel below
`sparse_loss_unfused_max_rows=4096` and never allocates its `d_model x vocab` fp32 accumulator. At
our loss density (0.0064 -> 104 rows) the gated head costs 2.1 ms.

### The launch-bound regime, and why graphs cannot be cashed in yet

MEASURED idle fraction and launch count (profiler, 4B/16k):

| arm | step ms | GPU-busy ms | **idle** | launches | graph upper bound |
|---|---|---|---|---|---|
| dense | 1420.3 | 1405.6 | **1.0%** | 6874 | 1.01x |
| `staticFFN+bskipK-0.25` | 546.7 | 513.5 | 6.1% | 7680 | 1.06x |
| `J-f1.00-k0.2500` | 357.2 | 300.2 | 16.0% | 10065 | 1.19x |
| `staticFFN+bskipK-0.25+headgate` | 228.0 | 191.7 | 15.9% | 6935 | 1.19x |
| `staticFFN+bskipK-0.0625+headgate` | 169.1 | 79.5 | **53.0%** | 6740 | **2.13x** |

The launch-bound regime is real but arrives late: ~1.19x at the recommended 6.4x point, 2.13x only
at the degenerate keep=0.0625. `graph upper bound` is `wall / GPU-busy` -- what a *perfect* graph
could recover, not a prediction.

**Can a training step be captured? A ladder, one case per process** (a failed capture poisons every
later capture in the same CUDA context). MEASURED:

| case | 0.8B / 2k | 4B / 16k |
|---|---|---|
| `matmul` (sanity) | captured, 1.04x | — |
| one block, forward | captured, **3.90x** | — |
| one block, fwd+bwd | captured, **4.38x** | — |
| whole model forward, no head | captured, 3.32x | — |
| whole model fwd+bwd, no head | captured, **2.65x** | captured, 1.003x |
| ... with `budget_attach` off | captured, 2.67x | — |
| whole step, **unfused** cross entropy | captured, **2.09x** | OOM (16k x 248k logits) |
| whole step, **fused-linear** CE (our trainer) | **FAILS** | **FAILS** |
| whole step, head gating as shipped | — | **FAILS** |
| whole step, **fixed-capacity** head gate (prototype) | — | **captured**, 1.006x |
| ... on any routed arm (`staticFFN+bskipK-*`, `J-f1.00-k0.25`) | — | **FAILS** |

The blockers are host synchronisations and value-dependent shapes, and they are enumerable:

1. **The fused-linear CE kernel.** Not capturable. Falls away automatically once the head is gated
   below `sparse_loss_unfused_max_rows`, because `LMHead` then uses the plain implementation.
2. **Head gating as shipped**: `lm_head.py:286` `bool(keep.any())` plus `x[keep]`, whose output
   shape depends on tensor *values*. Replacing it with a fixed row budget chosen by `topk`
   (`StaticGateHead` in `graph_probe.py`) makes the whole 4B/16k step capturable, and is **exact**
   for `capacity >= supervised rows`: surplus rows carry `ignore_index` and contribute zero loss
   and zero gradient, the same argument as fixed-capacity block skip.
3. **The block-skip holder's bookkeeping**, which is what still blocks every routed arm -- three
   host syncs: `block_skip.py:206` `float(keep.sum().item())` per routed block in `accumulate`,
   and `block_skip.py:152` `float(e.detach())` per routed layer plus `:166` `hist....tolist()` in
   `begin_forward`. A prototype removing only the first (`--bskip-nosync`, exact under fixed
   capacity) still fails to capture, which is how we know all three matter.
4. **The FFN router**: `nested_ffn_moe.py:799` `torch.bincount(choice, ...).tolist()` per routed
   layer.

None is load-bearing numerics -- they are budget statistics and dispatch counts that could live on
device and be read once per N steps. **That is the concrete engineering item**, and by the idle
column it is worth ~1.19x at the recommended operating point.

### So: is 20x reachable?

**Not in practice. 8.4x is the honest best measured, and the arithmetic to 20x needs GDN AND
attention to go to zero on top of it.** All MEASURED on one H200, batch 1, in one process against a
1419.0 ms dense control at 4B/16k:

| | step ms | speedup |
|---|---|---|
| dense | 1419.0 | 1.00x |
| **best all-learned (trainable)**, `J-f0.50-k0.0625` | 219.5 | **6.46x** |
| best trainable at a defensible keep, `J-f0.50-k0.2500` | 346.8 | 4.09x |
| **best with static FFN deletion** (post-training lever), `staticFFN+bskipK-0.0625+headgate` | 169.1 | **8.39x** |
| ... if GDN cost nothing (CALCULATED from its measured 92.9 ms) | 83.6 | 17.0x |
| ... if GDN and attention both cost nothing | 67.3 | 21.1x |
| the un-routable floor at that keep | 63.3 | **22.4x** |
| the target | 70.9 | 20x |

- **The ceiling is not one number -- it rises as block skip gets more aggressive**, because
  compaction removes skipped tokens from the block shell too: `dense/floor` is **11.6x** at dense
  and keep=0.5, **16.7x** at keep=0.25, **22.4x** at keep=0.0625. 27B/8k traces the same curve
  independently: **11.6x / 12.3x / 17.8x / 20.9x**. The earlier "8.7% floor implies 11.5x" is
  correct *at dense* and reproduces at both scales, but it is not a ceiling on an aggressively
  skipped model.
- **What actually stops us is GDN.** It is 53% of the best arm's step at 4B (34.8% at 27B, where
  it overtakes a 46.5% FFN) and it is *flat* from keep=0.25 to keep=0.0625 (92.9 ms both times) --
  skipping tokens stops helping it. Every GDN
  lever in this file is already refuted (fusion 1.02x, chunk size ~1-2%, packing 0.91x, expand_v
  ~1.4% of a step, more batch/head parallelism refuted); the section above concluded "stop
  optimising the GDN kernels", and this is that conclusion arriving as the binding constraint.
- **20x would need GDN at zero and attention at zero, at a keep of 1/16 that is not a trainable
  model.** Even a perfect GDN gives 17.0x.
- **Plan against 4-6.5x trainable and 5-8.4x with post-training FFN deletion, halved by packing**
  (4.22x / 5.08x measured at `--docs 32`). The next real 1.19x is CUDA graphs, blocked by four
  enumerable host syncs rather than by anything fundamental.
- **What changed versus the published 5.65x:** +13.6% from routing every FFN layer (6.42x for the
  same arm), and +31% beyond that from pushing keep 0.25 -> 0.0625 (8.39x), at an operating point
  that is a mechanism probe rather than a model. At 27B the same arms give 6.67x and 6.77x -- and
  27B saturates even harder (keep 0.25 -> 0.0625 buys 1.5%), with GPU busy falling to 34.4%.

**Artifacts for this section.** `joint_sweep_4B_16384_v1.json` (plain), `_packed32.json`
(`--docs 32`), `joint_sweep_27B_8192_v1.json` (AC), `graph_probe_*.json`. The block-shell
partition is `joint_sweep_4B_16384_shell.log` rather than a JSON -- that run filled the `/accounts`
quota at its final write, so its stdout is the record. **In that log, read the per-component
percentages only**: its `floor=` and `cap_from_here=` fields predate the fix that moved
`block_shell` into the floor sum, and are wrong (they report 0.3% / 382x for dense). The floor
figures quoted in the table above are recomputed as `step - (head waste + FFN + attn + GDN)`.
