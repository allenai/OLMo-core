# Molmo2 Stage-2: the fast single-image configuration

This branch (`donovan/stage2-fast-defaults`) makes the measured-fastest Stage-2 SFT
configuration the default, so a run started from `src/scripts/train/Molmo2-Stage2.py` on a
**single-image** tier gets roughly **2.1x** the training throughput (1.85x from the crop budget and pad fix, a further 1.13x from the attention block sizes) without anyone having
to rediscover the flags.

Everything below was measured on 8xB300 (`ai2/holmes`) against `single-image-only-v10`.

---

## What the config is

| Setting | Was | Now | Where |
|---|---|---|---|
| `pack_max_crops` (single-image tiers) | 25 | **80** | `SINGLE_IMAGE_PACK_MAX_CROPS` in `src/olmo_core/data/multimodal/mixtures/mixture_pack_profiles.py` |
| `pack_max_crops` (multi-image tiers) | 125 | **125** (unchanged) | same file |
| `DL_NUM_WORKERS` | 2 | **8** | `Molmo2-Stage2.py` |
| `RANK_MICROBATCH_INSTANCES` | 2 | **2** (unchanged, but load-bearing) | `Molmo2-Stage2.py` |
| `PYTORCH_CUDA_ALLOC_CONF` | unset on the Gantry path | **`expandable_segments:True`** | `SHIP_STACK_ENV` in `Molmo2-Stage2.py` |
| FlexAttention backward block sizes | Inductor defaults | **`BLOCK_M1/N1/M2/N2 = 32/128/128/32`** | `_FLEX_KERNEL_OPTIONS` in `nn/attention/backend.py` |

Plus the pad-attention fix (PR #865), cherry-picked onto this branch, worth a further
**+26-34% TPS** on its own; the occupancy metrics (PR #870), which are how you tell whether
*your* config is wasting compute; and the FlexAttention `kernel_options` tuning (PR #877),
worth **+12.65%** on its own.

### Why the attention block sizes are here

The LM's flex-attention backward was the largest single kernel in a profile at this config
-- 19.56% of GPU time and 3.8x its own forward, against a ~2.5x FLOP ratio. Inductor's
default block sizes are not tuned for the ~93%-sparse packed masks. Both arms run twice,
8xB300, 100 steps, medians past step 20, occupancy and memory identical throughout:

| config | runs | mean | vs control |
|---|---|---|---|
| Inductor defaults | 15,314 / 15,576 | 15,445 | — |
| **`kernel_options`** | **17,185 / 17,612** | **17,399** | **+12.65%** |

**Do not try to tune this with `mode="max-autotune..."` on the inner `torch.compile`.** It
measures -38.1% on the kernel in isolation and delivers **+2.17% end to end, inside the
control's own spread**: the call is inlined into the outer compiled LM block and the mode
is dropped. `kernel_options` is an argument, so it survives. See the note on
`_get_flex_attention`.

Nothing needs to be passed on the command line. A Gantry launch picks all of it up:

```bash
python src/scripts/train/Molmo2-Stage2.py launch <run-name> \
  --mixture=single-image-only-v10 \
  --launch.workspace=ai2/molmofication --launch.priority=high \
  --launch.preemptible=true --launch.clusters=[ai2/holmes] \
  --launch.beaker_image=akshitab/olmo-core-tch2100cu130-2026-07-03
```

### Why 80

A 16,384-token pack at the old 25-crop budget is ~69% padding: the pack fills its *crop*
budget (97.5% crop occupancy) long before it fills its *token* budget (30.6%), so the LM
runs 3.2x more padded 16k sequences than it needs to for the same ViT work. Total ViT crops
are exactly invariant across the whole sweep; only the number of LM sequences changes.

Swept at 100 steps (medians — two runs hit rare multi-hundred-second loader stalls that
wreck a mean), measured in *useful* (non-pad) TPS:

| crops | 25 | 32 | 40 | 50 | 64 | **80** | 100 |
|---|---|---|---|---|---|---|---|
| useful TPS | 8,400 | 9,527 | 11,164 | 12,140 | 14,815 | **15,729** | 14,624 |

80 is the peak. Token occupancy saturates at 98.7% there; 100 buys no additional occupancy
and costs step time (6.64s vs 6.17s) — crop budget with no tokens to spend it on is pure
cost.

### New metrics to read (PR #870)

- `throughput/device/useful TPS` — non-pad tokens/sec. **Use this, not raw TPS.** Raw TPS
  counts padded tokens, so it *rises* when a pack carries less real work; it reads every
  one of the wins above as a regression (`crops=50/mb=2` reads 18.3k padded TPS against the
  baseline's 27.2k).
- `throughput/device/token occupancy` — fraction of the 16k pack that is real. 30.8% before,
  98.7% now. If yours is low, you are paying for padding.
- `throughput/device/crop occupancy` — ~97%. Known ~2pp optimistic: it is measured on the
  collator's rank-local batch and misses the DP-wide crop padding `_encode_images` adds.

---

## Measured speedup

2,000 steps, 8xB300, `single-image-only-v10`:

| | rows/sec (8 GPUs) | speedup per row |
|---|---|---|
| `crops=25` (previous default) | 54.0 | — |
| `crops=64` | 91.3 | 1.69x |
| **`crops=80`** | **100.0** | **1.85x** |

- Useful TPS **15,729 / 15,785** (n=2, 0.36% spread) vs **8,400** baseline.
- Token occupancy **30.8% -> 98.7%**; crop occupancy ~97%.
- A production run of **15.7M rows** goes from **80.8 h to 43.7 h** — ~37 hours saved.
  At the more conservative `crops=64`, 47.8 h.
- **16 GPUs: 187.0 rows/sec = 1.87x for 2x the hardware** (94% scaling), per-device useful
  TPS **14,715 averaged over a completed 10,000-step run** (median of the per-step samples:
  15,045). The average is the honest production number — it absorbs the checkpoint saves
  (every 1,000 steps, plus ephemeral every 250) that a 100-step smoke never pays.
- **Memory:** peak reserved **232.9 GiB of 267.7 — 34.8 GiB headroom (13%)**. The allocator
  arena grows for the first ~300 steps after start, reaches 232.9 GiB at **step 5,060**, and
  is then *exactly* flat for the remaining **4,940 steps**. Nothing drifts; but note this
  ceiling is ~10 GiB above what the run showed before it got there, so **do not read a peak
  off a run that has not yet plateaued** (an earlier reading of 223.0 GiB, taken at step
  4,874, was a pre-plateau undercount).

**`reduce_dtype=bfloat16` was tested here and rejected.** It is convergence-neutral
(largest binned CE delta 0.0008 over 500 steps against a within-bin std of ~0.03) but
**-1.50% slower** at 500 steps. An earlier 100-step pair suggested +5.18%; that was noise
from a single unreplicated comparison. Do not re-derive it.

**Convergence is neutral.** CE against *examples consumed* (the fair axis: `crops=80` sees
1.26M examples in 2,000 steps vs the baseline's 1.05M), binned: final-bin means differ by
**0.0011** against a within-bin std of ~0.027. Do not read raw end-of-run CE (0.8128 /
0.7642 / 0.8028) — those are single noisy steps and point the wrong way.

---

## What is NOT proven

Read this section before treating any of the above as settled.

1. **n=1 for the long run.** The 10,000-step proof run **completed** (Beaker
   `01M2RR6GNY7WZS5YN291Z85Y4K`, 16xB300, final CE 0.674, both nodes exit 0), which is the
   strongest stability evidence here — but it is one observation, on a configuration that
   has already hidden a tail failure once (see the allocator trap below). The 100-step and
   2,000-step numbers repeat well; the multi-thousand-step behaviour does not.
   It also ran as **two segments**: preempted at step 4,874, resumed from `step4750`. A
   restart re-initialises the allocator, so the run does not prove that a *single
   uninterrupted* 10,000-step process holds 232.9 GiB — only that each segment does.
3. **Single-image tier only.** The multi-image profile (`pack_max_crops=125`) is *untested*
   at these settings and deliberately unchanged. A multi-image row costs several images'
   worth of crops, so none of the single-image occupancy arithmetic transfers.
4. **Convergence was checked over 2,000 steps, not to convergence.** "Indistinguishable CE
   against examples consumed over 2,000 steps" is not "same final model quality". No eval
   comparison of a fully-trained `crops=80` checkpoint against a `crops=25` one exists.

**If you want margin, use `crops=64`** (+69.9% instead of +85%): more memory headroom, and
it does not sit on the one long run above. `--pack_max_crops=64` overrides the profile.
This matters more now than it did at the 223 GiB reading: 34.8 GiB of headroom is 13%, so a
mixture whose crop distribution runs heavier than `single-image-only-v10`'s has less room
than the earlier number suggested.

---

## Operational traps

### 1. Batch granularity — the thing most likely to bite you

`GLOBAL_BATCH_INSTANCES` counts **packs**, not examples, and a pack now holds **~13.1
examples instead of ~4.1**. Leaving it at the old 128 would have meant **~1,677
examples/step** instead of ~524 — a 3.2x batch change smuggled in alongside a performance
change, against an unchanged LR schedule.

**The default is therefore 32 packs (~419 examples/step)**, slightly *below* the old
baseline, so adopting this config does not silently enlarge your batch. `crops=80` was
measured at 32 packs and still gave **+85.6%** (vs +87.3% at 48), so the speedup does not
depend on a large batch.

Global packs must divide `dp_world_size x rank_microbatch_instances` — **16** at 8 GPUs
with mb=2, **64** at 32 GPUs:

| packs | examples/step | 8 GPU | 16 GPU | 32 GPU | notes |
|---|---|---|---|---|---|
| **32** | ~419 | OK | OK | **FAIL** | the default; validated at +85.6% |
| 48 | ~629 | OK | FAIL | FAIL | what most of the sweep ran at |
| 64 | ~838 | OK | OK | OK | smallest valid choice at 32 GPUs |
| 128 | ~1,677 | OK | OK | OK | the old default; 3.2x the old batch, unvalidated |

**So a 32-GPU run must raise it** — 64 packs is the smallest that works there.
`launch()` checks this against the requested GPU count and refuses on your machine, naming
the valid values, rather than letting the trainer abort after the job has been scheduled.
Matching the old 524 examples/step exactly would need ~40 packs, which is unreachable at
mb=2 on any of these world sizes.

### 2. `expandable_segments` is not optional

Without `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `crops=80` **OOMs at around
step 1,307** — *after passing three separate 100-step smokes* — reporting ~39 GiB
reserved-but-unallocated. **A 100-step smoke cannot validate memory stability at this crop
budget.**

It is now in `SHIP_STACK_ENV`, so every `Molmo2-Stage2.py launch` gets it. `SHIP_STACK_ENV`
reaches Beaker launches only, so if you run under raw `torchrun`, **export it yourself**.
The script logs a warning at step 0 if you do not.

### 3. `mb=1` is not a safe fallback

`RANK_MICROBATCH_INSTANCES=1` costs **~19%** at identical occupancy and examples/step
(10,204 vs 12,140 useful TPS at `crops=50`). `crops=80` at mb=2 does **not** OOM — the
assumption that it would, untested, once cost half the measured win. `mb=3` does OOM on
B300 (~261/268 GiB).

### 4. `DL_NUM_WORKERS=2` starves the loader

At the raised crop budget a fuller pack is more packing work per step, and the packer runs
inside these workers. At 2, the run measured **43% data-loader-bound**. 8 fixes it; 16 is
indistinguishable from 8 (12,146 vs 12,140), so there is nothing above 8 to buy.

### 5. `pack_max_crops` is not an eval `max_crops`

`pack_max_crops` is the *packer's per-pack capacity*, unrelated to the per-image crop
budget (`MAX_CROPS = 8`) the model was trained with. A stage-2 `config.json` records only
`pack_max_crops`, and tooling has previously derived an eval `max_crops` from it. With this
default that would pass **80** to the eval harness, which is wrong. The current image-QA
eval scripts hardcode `max_crops=24` and are unaffected — but check before evaluating a
checkpoint trained on this branch.

---

## Branch provenance — this stacks three unlanded PRs

| PR | What | Commits here |
|---|---|---|
| **#834** | `image-only-v10` Stage-2 port: the `single-image-*` tiers, `mixture_pack_profiles.py`, `DL_NUM_WORKERS`, `SHIP_STACK_ENV` | merged in (`Merge the image-only-v10 Stage-2 branch`) |
| **#865** | pad-attention fix (+26-34% TPS) | 3 cherry-picked commits |
| **#870** | `useful TPS` + token/crop occupancy metrics | 3 cherry-picked commits |
| **#877** | FlexAttention `kernel_options` (+12.65%) | `eadd38648`, `48ab6db18` |

All three are still under review. **If any of them changes, rebase this branch onto the new
version before using it** — particularly #865, whose attention-mask semantics the throughput
numbers above depend on.

`origin/vision` does not contain #834, so none of the settings in this document are even
expressible there: there is no `single-image-*` tier, `pack_max_crops` is a single global
125 rather than tier-derived, and there is no `DL_NUM_WORKERS` knob or `SHIP_STACK_ENV`.
That is why the merge is here rather than the branch being a three-commit diff on `vision`.
