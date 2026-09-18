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
