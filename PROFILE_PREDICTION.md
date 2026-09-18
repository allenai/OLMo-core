# Pre-registered prediction for the crops=80 profile

Written **before** the profile job (`profile-crops80-8gpu`) returned, so the
interpretation cannot be retrofitted to whatever came back.

## The arithmetic

Per 16,384-token pack, moving `pack_max_crops` 25 -> 80:

| | crops=25 | crops=80 | ratio |
|---|---|---|---|
| ViT crops | ~24 | ~79 | **3.3x** |
| real LM tokens | ~5,046 (30.8%) | ~16,180 (98.8%) | 3.2x |
| *padded* LM tokens | 16,384 | 16,384 | **1.0x** |

The last row is the whole point. QKV projections, MLP, norms and RoPE run on the full
padded 16,384 regardless of occupancy -- only attention benefits from pack sparsity (flex
skips masked blocks). So **LM dense FLOPs per pack are unchanged** by the crop budget,
while **ViT FLOPs per pack tripled**.

Taking the last recorded split (~28% ViT at crops=25) and holding LM dense constant:

- ViT: 28 -> 28 x 3.3 = **92**
- LM + everything else: **~72** (plus a modest rise in attention, which was the sparse part)
- ViT share: 92 / ~164 = **~56%**

Vision activation checkpointing is ON (`multimodal_train_module.py:84`, never overridden,
and turning it off OOMs at every useful crop budget). So the ViT pays forward + recompute
forward + backward ~ 4x forward, against the LM's forward + backward ~ 3x forward
(`ac_config=null`). That pushes the ViT's share of *time* above its share of FLOPs,
plausibly to **~60%**.

**Independent check that the arithmetic is not nonsense.** Per-pack step time measured:
crops=25 is 9.62 s / 16 packs per rank = 0.60 s/pack; crops=80 is 6.17 s / 6 = 1.03 s/pack.
Ratio **1.71x**, against a predicted FLOP ratio of 164/100 = **1.64x**. These agree to 4%,
which they had no obligation to do.

## Predictions

1. **The step is now ViT-dominated, ~50-60% of GPU time** (was ~28%). If this holds, the
   ranking that has driven this work for months is inverted, and ViT-side levers --
   including **fp8/MXFP8 on the ViT, which was scoped out of this round** -- become the
   highest-value remaining work rather than a footnote.
2. **The FSDP/NCCL share has fallen sharply** from the recorded ~45%. Collectives are
   per-parameter and roughly constant per step, while compute per step rose ~1.7x, so the
   same absolute comm time is a much smaller fraction. Concretely: if comms were ~45% at
   0.60 s/pack, the same absolute time at 1.03 s/pack is **~26%**.
   **This would demote Phase 4 (collectives) from "next up" to "probably not worth it"**,
   since its three levers split a shrinking pie.
3. **fp32 `reduce_scatter`, previously 26.8% of CUDA time, is now ~15%** by the same
   argument -- which also weakens the case for re-testing `reduce_dtype=bfloat16` at
   8 GPUs (it already regressed 4.8% there). It does *not* weaken the 32-GPU case, where
   the reduce goes off-node.

## What would falsify this

- ViT under ~40% of GPU time. That would mean the crop-count -> FLOP scaling is not
  translating into time (e.g. the ViT is launch-bound or memory-bound at these sizes, not
  FLOP-bound), and the chunking in `_encode_images` becomes the thing to look at instead.
- FSDP/NCCL still at ~45%. That would mean comm time is *not* constant per step -- most
  likely because it is bandwidth-bound on activations/gradients that grew with the crop
  count, not just on parameters. Phase 4 would stay live.

---

# CORRECTION, written before the profile returned

The prediction above is **wrong**, and I am leaving it in place rather than editing it.
It was superseded by a better calculation, not by data -- the profile job was still
`QUEUED` when this was written.

## What was wrong

Prediction 1 rested on a recorded "~28% ViT at `crops=25`" figure. That number is a
**measured time share** from an old profile. I treated it as a FLOP share and scaled it by
the crop ratio. Computing the FLOP split properly from the real `molmo2_4B` config
(`num_flops_per_token` and `image_encoder_flops`, meta device, no weights) gives something
very different:

| crops/pack | ViT mode | LM PF | ViT PF | connector PF | **ViT share of FLOPs** |
|---|---|---|---|---|---|
| 24 | frozen | 0.8704 | 0.0149 | 0.0012 | 1.7% |
| 24 | trained | 0.8704 | 0.0446 | 0.0012 | 4.9% |
| 79 | frozen | 0.8704 | 0.0489 | 0.0039 | 5.3% |
| **79** | **trained** | **0.8704** | **0.1467** | **0.0039** | **14.4%** |

**The LM dominates FLOPs even at `crops=80` -- roughly 6:1.** The ViT runs 57.6k patch
tokens through a 0.38B encoder; the LM runs 16,384 tokens through a 4B model with a
quadratic attention term at S=16384. The ViT was never going to be half the step.

The "1.64x predicted vs 1.71x measured" agreement I cited was therefore **numerology** --
two wrong inputs landing near a right answer. The real per-pack FLOP ratio is 1.11x, not
1.64x.

## What the corrected arithmetic says instead

| | crops=25 | crops=80 |
|---|---|---|
| PF per pack | 0.916 | 1.021 |
| packs per rank per step | 16 | 6 |
| **total FLOP throughput** | 1.524 PF/s | **0.993 PF/s (-35%)** |
| **useful LM FLOP throughput** | 0.446 PF/s | **0.836 PF/s (+87%)** |

This is the honest account of the win, and it is a sharper story than the one I had:
**`crops=80` makes the hardware do *less* raw work per second, not more.** Absolute FLOP
throughput falls 35%. It wins because at `crops=25` most of the LM's FLOPs were spent on
padding, and the +87% in *useful* LM FLOPs/s lands right on top of the independently
measured +85-87% useful TPS and 1.85-1.87x rows/sec.

## Revised predictions

1. **The LM dominates the profile, not the ViT.** `mm::lm_forward` should be the large
   majority of GPU time; `mm::vision_encode` should be well short of half.
2. **But the ViT should punch above its 14.4% FLOP weight.** The 35% drop in raw FLOP
   throughput has to come from somewhere, and the ViT is the component whose share grew.
   If `mm::vision_encode` comes in at, say, 25-35% of step time against 14.4% of FLOPs,
   that gap -- not the FLOP count -- is the remaining ViT lever, and it would point at
   launch/memory-bound behaviour in `_encode_images` chunking rather than at fp8.
3. **Prediction 2 from the original (comms share falls) still stands**, and for a reason
   the correction does not touch: comms are per-parameter and roughly constant per step,
   while per-step time rose. I expect FSDP/NCCL well below the recorded ~45%.

## Falsification, restated

- `mm::vision_encode` at ~14% of time would mean the ViT is running at the same efficiency
  as the LM and there is no ViT lever at all -- the 35% throughput drop would then have to
  be comms or attention, and Phase 4 comes back to life.
- `mm::lm_forward` under half the step would falsify revised prediction 1 outright.

## Fallout: reported MFU is an undercount on this branch

`image_encoder_flops` charges the ViT **forward-only** on an explicit "the encoder is
frozen" docstring assumption. Stage 2 **trains** the ViT (`VISION_LR=5e-6`), so the term
should be 3x. The fix (`vision_is_trainable()`) exists on `donovan/perf-deep-dive` but is
**not** on `donovan/stage2-fast-defaults`, which is the branch intended for others to use.

At `crops=80` this understates total FLOPs by **10.6% relative**: the proof run's reported
**41.35% MFU is really ~45.7%**. At `crops=25` the error is only 3.4%, so the bug also
makes the crop-budget change look slightly worse than it is on the MFU axis. Worth porting.

---

# SECOND CORRECTION (still before the profile returned)

The first correction was right that the ViT is ~14% of *FLOPs*. It was wrong to conclude
from that that the ViT is a small share of *time*. Separating the LM's dense and attention
terms and solving for each component's throughput changes the picture again.

## The LM's attention term is mostly charged, not computed

`num_flops_per_token(16384)` decomposes as:

| | FLOPs/token | share |
|---|---|---|
| dense (params) | 2.4134e10 | 45.4% |
| attention (quadratic in S) | 2.8991e10 | **54.6%** |

That attention term assumes **dense causal attention over the whole 16,384-token pack**.
With packing plus the pad fix, attention only computes within-example causal blocks -- at
~13 examples of ~1,245 tokens, that is `sum(len_i^2)` against `S^2`, i.e. a small fraction.
Actually-computed attention is **0.0356 PF/pack against 0.475 PF charged, ~13x less.**

So MFU is **overstated on the attention axis** at the same time as it is understated on the
ViT axis. Charged FLOPs per pack at crops=80 are ~1.017 PF against ~0.582 PF actually
computed. This is the "MFU convention reconciliation" item, and it is larger than the
trainable-ViT fix -- but it is a *convention* question (the standard MFU formula charges
dense attention), not a bug, so it should be reported alongside the conventional number
rather than silently replacing it.

## Solving for per-component throughput

Charging only what is computed:

| arm | LM dense | LM attn | ViT | total PF |
|---|---|---|---|---|
| crops=25 | 0.3954 | 0.0113 | 0.0457 | 0.4524 |
| crops=80 | 0.3954 | 0.0356 | 0.1505 | 0.5816 |

Fitting `step_time = LM_flops / r_lm + ViT_flops / r_vit` across the two arms:

- implied **LM throughput 0.960 PF/s**
- implied **ViT throughput 0.259 PF/s** -- the ViT is **3.7x less FLOP-efficient**

| arm | ViT share of FLOPs | **predicted ViT share of time** |
|---|---|---|
| crops=25 | 10.1% | **29.4%** |
| crops=80 | 25.9% | **56.4%** |

The 29.4% at crops=25 is a genuine check: it reproduces the independently recorded ~28%
ViT time share, which was not an input to the fit.

**This is a two-parameter model fitted to two measured step times -- zero degrees of
freedom.** It is a consistency construction, not evidence. It cannot fail to fit. Treat it
as "what the numbers imply if each component has a characteristic throughput", and let the
profile decide.

## Where this leaves the prediction

My original 50-60% ViT was **right on the number and wrong on the reason**. It is not that
the ViT has half the FLOPs (it has ~26% of the computed ones); it is that the ViT converts
FLOPs to time ~3.7x worse than the LM does. The first correction over-rotated.

**Final prediction: `mm::vision_encode` lands at 45-60% of step time at crops=80.**

The actionable consequence is unchanged and now better supported: the ViT lever is an
*efficiency* lever, not a FLOP-reduction lever. A 3.7x gap is not something fp8 closes
(fp8 would cut ViT FLOPs, which are not the problem) -- it points at `_encode_images`
chunking, small per-crop GEMMs, or launch overhead across 25 small blocks.

Falsifier: `mm::vision_encode` at or near 26% would mean the ViT runs at LM-like
efficiency, the 3.7x is an artifact of the two-point fit, and the remaining time is
elsewhere -- most likely comms, which would revive Phase 4.
