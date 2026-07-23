# Concrete Architecture Runbook Summary

This document is the short-form index for the concrete post-baseline MoE A0
experiments. It captures the current launch intent, LR-transfer assumptions,
systems defaults, and first-wave variants. The per-experiment runbooks remain
the source of truth for exact implementation details.

## Shared Policy

Run these experiments only after the relevant baseline rungs are sufficiently
settled for comparison. For now, LR selection uses final-window training CE
only; validation/eval metrics are logged for later analysis but are not used to
choose LRs.

Default LR policy:

1. Start with three powers-of-two-spaced LRs centered on the corresponding
   baseline fitted optimum.
2. Use the Cx1 result to estimate an architecture-specific LR multiplier:

   ```text
   m_variant = lr*_variant_Cx1 / lr*_baseline_Cx1
   ```

3. Center later rungs on `m_variant * lr*_baseline_CxN`.
4. If the best point lands on an edge, add one powers-of-two extension in the
   improving direction.
5. If the curve is bracketed, fit a 3-point local quadratic in log LR and report
   both best observed LR and fitted optimum.

Default systems policy:

| Size | Cx1 | Cx2 | Cx4 | Cx8 |
| --- | ---: | ---: | ---: | ---: |
| 275M | 1 GPU | 2 GPUs | 4 GPUs | 8 GPUs |
| mid_480m | 4 GPUs | 4 GPUs | 4 GPUs | 8 GPUs |
| 810M | 8 GPUs | 8 GPUs | 8 GPUs | 16 GPUs |
| 1.2B | 8 GPUs | 8 GPUs | 16 GPUs | 32 GPUs |

Use EP=1 by default. Smoke test any variant that changes memory materially
before queueing full training. Use compact run-name tags and keep batch/resource
details in trackers rather than expanding every run name.

Default checkpoint/eval flags:

```text
--ladder-evals
--eval-task-set=fast
--eval-interval=2000
--save-interval=999999999
--ephemeral-save-interval=500
--no-pre-train-checkpoint
```

TODO: Decide later how validation losses should influence LR choice, promotion,
and checkpoint selection. Current policy is training-loss-only for LR selection.

## 1. Expert Granularity

Runbook: `expert_granularity.md`

Question: at fixed routed active capacity and fixed routed total capacity,
should experts be fewer/larger or more/smaller?

Invariant family:

```text
num_experts / top_k = 12
top_k * moe_hidden_size = 4 * d_model
num_experts * moe_hidden_size = 48 * d_model
```

Variants:

| Variant | Experts | `top_k` | `moe_hidden_size` | Initial role |
| --- | ---: | ---: | ---: | --- |
| `coarse_24e_top2` | 24 | 2 | `2 * d_model` | Full 275M ladder. |
| `baseline_48e_top4` | 48 | 4 | `d_model` | Existing control. |
| `fine_96e_top8` | 96 | 8 | `d_model / 2` | Full 275M ladder. |
| `extreme_192e_top16` | 192 | 16 | `d_model / 4` | 275M Cx1 diagnostic probe. |
| `ultra_384e_top32` | 384 | 32 | `d_model / 8` | 275M Cx1 diagnostic probe. |

Initial launch scope:

- Run the whole 275M ladder for `coarse_24e_top2` and `fine_96e_top8`:
  Cx1/Cx2/Cx4/Cx8/Cx16.
- Use three LRs per rung. Current first-wave centers are:
  - Cx1: `1e-3`, `2e-3`, `4e-3`
  - Cx2: `5e-4`, `1e-3`, `2e-3`
  - Cx4: `8e-4`, `1.6e-3`, `3.2e-3`
  - Cx8: `8e-4`, `1.6e-3`, `3.2e-3`
- Run `extreme_192e_top16` and `ultra_384e_top32` as Cx1-only probes before
  considering promotion.

Promotion:

- If a serious variant beats baseline cleanly, promote it to 810M Cx1/Cx4.
- If the win holds at 810M, consider 1.2B and/or additional Cx rungs.
- If an extreme variant wins Cx1, confirm with Cx4 before treating it as a
  serious architecture candidate.

## 2. Total Sparsity At Fixed Active Compute

Runbook: `total_sparsity_fixed_active.md`

Question: if each token activates the same routed compute, does adding inactive
expert capacity help?

Keep fixed:

- `top_k = 4`
- `moe_hidden_size = d_model`
- shared expert / dense prefix / attention schedule

Vary `num_experts`, and therefore total parameters.

Initial variants:

| Variant | Experts | `top_k` | Approx active/total intent |
| --- | ---: | ---: | --- |
| `baseline_48e_top4` | 48 | 4 | Existing control. |
| `high_total_96e_top4` | 96 | 4 | More total capacity, roughly modern-small-MoE sparse. |
| `huge_total_192e_top4` | 192 | 4 | Aggressive sparse point, near the low end of public active/total ratios. |

Hold `low_total_24e_top4` for later. Our baseline is already not especially
sparse compared with recent public MoEs, so the first wave should move toward
more total capacity rather than less.

Initial launch scope:

- 275M Cx1 and Cx4 for `high_total_96e_top4` and `huge_total_192e_top4`.
- Run three LRs per rung centered on the baseline optimum / transferred Cx1
  multiplier.
- Promote to 275M Cx8 if either variant improves or ties with acceptable
  throughput.
- Promote at most one total-sparsity variant to 810M Cx1/Cx4 initially.

Public sparsity anchors:

- Recent public MoEs commonly activate roughly 3-17% of total parameters.
- The smaller current public points are often around 9-16% active/total, while
  frontier-scale systems can be closer to 3-5%.
- This makes `96E/top4` and `192E/top4` reasonable first targets; an even more
  sparse setting can wait until we know whether this direction helps.

## 3. Shared Expert And Dense Schedule

Runbook: `shared_expert_dense_schedule.md`

Question: is the shared expert helpful, and can extra dense layers or dense/MoE
interleaving replace it?

Baseline:

```text
dense_prefix_layers = 1
num_shared_experts = 1
shared_mlp_hidden_size = d_model / 2
moe_hidden_size = d_model
```

First-wave variants:

| Variant | Dense schedule | Shared expert | Routed hidden | Initial role |
| --- | --- | --- | ---: | --- |
| `1dense_no_shared_am` | Dense layer 0 only | none | `9/8 * d_model` | Shared-expert ablation. |
| `2dense_no_shared_am` | Dense layers 0-1 | none | `9/8 * d_model` | Dense-prefix replacement. |
| `alternating_no_shared_am` | Dense even layers | none | `9/8 * d_model` | Dense/MoE interleaving probe. |
| `alternating_shared` | Dense even layers | `d_model / 2` | `d_model` | Optional follow-up only. |

The no-shared variants are active-matched, not total-matched:

```text
baseline active MLP hidden = 4 * d_model + d_model / 2
no-shared active MLP hidden = 4 * (9/8 * d_model)
```

Initial launch scope:

- 275M Cx1/Cx4 for the three no-shared first-wave variants.
- Run three LRs per rung centered on transferred baseline optima.
- Add `alternating_shared` only if interleaving looks promising or ambiguous.
- Promote the best variant to 810M Cx1/Cx4 after 275M confirmation.

## 4. Width / Depth Geometry

Runbook: `width_depth_geometry.md`

Question: at roughly fixed active and total parameter count, does this MoE
family prefer deeper/narrower or shallower/wider backbones?

Initial 275M candidates:

| Variant | `d_model` | `d_attn` | Layers | Heads | KV heads |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline` | 768 | 1024 | 12 | 8 | 4 |
| `deep_narrow` | 640 | 768 | 16 | 6 | 3 |
| `shallow_wide` | 896 | 1280 | 9 | 10 | 5 |

Initial launch scope:

- 275M Cx1/Cx4 for `deep_narrow` and `shallow_wide`.
- Run dry parameter checks first. Keep active/total params within roughly 5% of
  the baseline size if possible.
- Run three LRs per rung centered on transferred baseline optima.
- Promote the winning direction to 810M Cx1/Cx4, after finalizing the 810M
  layer count with dry checks.

Still to settle before implementation:

- Final 810M `deep_narrow` layer count.
- Final 1.2B `shallow_wide` layer count.
- Whether the odd KV-head counts in some candidates are acceptable or should be
  replaced with stricter head shapes.

## 5. SWA / Full Attention Ratio

Runbook: `swa_full_attention_ratio.md`

Question: how much full attention does the model need, and can we move toward
more local attention without hurting loss/evals?

Current control:

```text
pattern = [WINDOW_SIZE, -1]
force_full_attention_on_first_layer = False
force_full_attention_on_last_layer = True
```

This is effectively 1 SWA : 1 full attention, with final full attention.

Initial variants:

| Variant | Schedule | Motivation |
| --- | --- | --- |
| `attn_3to1` | 3 SWA : 1 full | Intermediate efficiency point. |
| `attn_5to1` | 5 SWA : 1 full | Gemma/MAI-style local/global ratio. |
| `attn_swa_final_full` | SWA everywhere except final full | Extreme sentinel. |

Do not run a denser-than-current attention schedule in the first wave.

Initial launch scope:

- 275M Cx1/Cx4 for `attn_3to1`, `attn_5to1`, and
  `attn_swa_final_full`.
- Run three LRs per rung centered on transferred baseline optima.
- Treat throughput as part of the result, not just loss.
- Promote a sparse-attention variant only if it matches/improves training loss
  and does not show an obvious validation/eval regression.

Public attention anchors:

- Current baseline and GPT-OSS-style sparse attention are roughly 1:1
  local/full.
- Gemma 3 and MAI-Thinking-1 use a 5 local : 1 global style schedule.
- Qwen3-Next is hybrid rather than a pure SWA/full control, but its 3 cheap
  layers : 1 attention-layer rhythm motivates the intermediate `attn_3to1`
  point.

## Plotting And Organization

Each experiment should get its own output directory for scripts, plots, and
bookkeeping once implemented:

```text
src/scripts/train/jacobm_olmoe_ladder/experiments/<experiment_name>/
```

Recommended plot structure:

- per-model ladder plots, with baseline and variants clearly separated;
- per-Cx cross-model plots, sorted by model size;
- per-experiment comparison plots at selected/best LR;
- U-plots for every rung where an LR curve was run;
- optional x-axis plots for structural sweeps:
  - expert granularity: experts/top_k or expert hidden size;
  - total sparsity: active/total percentage;
  - attention ratio: full-attention layer fraction;
  - width/depth: layer count or width/depth ratio.

Do not mix experimental variants into baseline-only plots without explicit
labels and separate output paths.
