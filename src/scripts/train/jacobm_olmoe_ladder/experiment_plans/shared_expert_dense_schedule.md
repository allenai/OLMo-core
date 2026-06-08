# Shared Expert And Dense Schedule Experiment Plan

This runbook specifies the shared-expert and dense/MoE layer-schedule ablation.
It is designed to run after the baseline ladder is settled and preferably after
the expert-granularity experiment identifies whether the current 48E/top4 block
remains the standard MoE geometry.

## Status

Planned, not yet implemented or launched.

Do not launch until:

- the current baseline has stable LR centers for the target Cx rungs;
- the expert-granularity plan has either completed or Jacob decides this axis is
  more urgent;
- we have decided whether to use the baseline 48E/top4 geometry or a new
  granularity winner as the default MoE block.

## Goal

Understand whether the current shared expert is genuinely useful, and whether
its usefulness depends on having only one dense prefix layer.

Current baseline:

```text
dense_prefix_layers = 1
num_shared_experts = 1
shared_mlp_hidden_size = d_model / 2
MoE layers after the dense prefix
```

## Matching Policy

Changing shared expert size cannot keep both active and total expert capacity
exactly fixed if `num_experts`, `top_k`, and routed `moe_hidden_size` are all
held fixed. For this experiment, the primary no-shared variants are therefore
**active-matched**:

```text
baseline active MLP hidden = 4 * d_model + d_model / 2 = 4.5 * d_model
no-shared active MLP hidden = 4 * moe_hidden_size
moe_hidden_size = 9/8 * d_model
```

This matches active MLP capacity per MoE layer while increasing routed total
expert capacity. Treat the total-parameter increase as part of the no-shared
active-matched design and report it explicitly.

Always report active and total params for each variant. Do not claim exact
fixed-total matching for this experiment.

## Variants

### Core Configs

Use the current baseline 48E/top4 expert geometry for shared variants. Use
active-matched routed hidden size for no-shared variants.

| Variant | Dense schedule | Shared expert | Routed hidden | Purpose |
| --- | --- | --- | --- | --- |
| `baseline_1dense_shared` | Dense layer 0 only | `d_model / 2` | `d_model` | Current control. |
| `1dense_no_shared_am` | Dense layer 0 only | none | `9/8 * d_model` | Isolates shared expert removal while active-matching. |
| `2dense_no_shared_am` | Dense layers 0-1 | none | `9/8 * d_model` | Tests whether one extra dense prefix helps replace shared expert. |
| `alternating_no_shared_am` | Dense even layers, MoE odd layers | none | `9/8 * d_model` | Tests dense/MoE interleaving without shared expert. |
| `alternating_shared` | Dense even layers, MoE odd layers | `d_model / 2` | `d_model` | Optional follow-up if interleaving is promising. |

This is too many to launch all at once. First launch:

1. `1dense_no_shared_am`
2. `2dense_no_shared_am`
3. `alternating_no_shared_am`

Keep `baseline_1dense_shared` as the existing control. Add `alternating_shared`
only if `alternating_no_shared_am` looks promising or ambiguous.

## Exact Baseline Fields By Size

Use these as fixed backbone fields unless the plan is updated:

| Size | `d_model` | `d_attn` | Layers | Heads | KV heads | Experts | `top_k` | `moe_hidden_size` | Shared hidden | Dense MLP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 275M | 768 | 1024 | 12 | 8 | 4 | 48 | 4 | 768 | 384 | 3456 |
| 810M | 1280 | 1536 | 20 | 12 | 6 | 48 | 4 | 1280 | 640 | 5760 |
| 1.2B | 1536 | 2048 | 22 | 16 | 8 | 48 | 4 | 1536 | 768 | 6912 |

For no-shared primary variants:

```text
num_shared_experts = 0
shared_mlp_hidden_size = 0
moe_hidden_size = 9/8 * d_model
dense_layer_mlp = 9/2 * d_model
```

For shared variants:

```text
num_shared_experts = 1
shared_mlp_hidden_size = d_model / 2
dense_layer_mlp = 4 * d_model + shared_mlp_hidden_size
```

### Exact 275M Configs

| Variant | `d_model` | `d_attn` | Layers | Dense layers | Experts | `top_k` | `moe_hidden_size` | Shared hidden | Dense MLP |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `baseline_1dense_shared` | 768 | 1024 | 12 | `[0]` | 48 | 4 | 768 | 384 | 3456 |
| `1dense_no_shared_am` | 768 | 1024 | 12 | `[0]` | 48 | 4 | 864 | 0 | 3456 |
| `2dense_no_shared_am` | 768 | 1024 | 12 | `[0, 1]` | 48 | 4 | 864 | 0 | 3456 |
| `alternating_no_shared_am` | 768 | 1024 | 12 | `[0, 2, 4, 6, 8, 10]` | 48 | 4 | 864 | 0 | 3456 |
| `alternating_shared` | 768 | 1024 | 12 | `[0, 2, 4, 6, 8, 10]` | 48 | 4 | 768 | 384 | 3456 |

### Exact 810M Configs

| Variant | `d_model` | `d_attn` | Layers | Dense layers | Experts | `top_k` | `moe_hidden_size` | Shared hidden | Dense MLP |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `baseline_1dense_shared` | 1280 | 1536 | 20 | `[0]` | 48 | 4 | 1280 | 640 | 5760 |
| `1dense_no_shared_am` | 1280 | 1536 | 20 | `[0]` | 48 | 4 | 1440 | 0 | 5760 |
| `2dense_no_shared_am` | 1280 | 1536 | 20 | `[0, 1]` | 48 | 4 | 1440 | 0 | 5760 |
| `alternating_no_shared_am` | 1280 | 1536 | 20 | even layers | 48 | 4 | 1440 | 0 | 5760 |
| `alternating_shared` | 1280 | 1536 | 20 | even layers | 48 | 4 | 1280 | 640 | 5760 |

### Exact 1.2B Configs

| Variant | `d_model` | `d_attn` | Layers | Dense layers | Experts | `top_k` | `moe_hidden_size` | Shared hidden | Dense MLP |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| `baseline_1dense_shared` | 1536 | 2048 | 22 | `[0]` | 48 | 4 | 1536 | 768 | 6912 |
| `1dense_no_shared_am` | 1536 | 2048 | 22 | `[0]` | 48 | 4 | 1728 | 0 | 6912 |
| `2dense_no_shared_am` | 1536 | 2048 | 22 | `[0, 1]` | 48 | 4 | 1728 | 0 | 6912 |
| `alternating_no_shared_am` | 1536 | 2048 | 22 | even layers | 48 | 4 | 1728 | 0 | 6912 |
| `alternating_shared` | 1536 | 2048 | 22 | even layers | 48 | 4 | 1536 | 768 | 6912 |

## Required Code Changes

Add a dense/shared schedule option to
`src/scripts/train/jacobm_olmoe_ladder/tiny_275m.py`.

Recommended CLI:

```text
--dense-schedule {baseline_1dense_shared,1dense_no_shared_am,2dense_no_shared_am,alternating_no_shared_am,alternating_shared}
```

Implementation notes:

- Current script raises unless `dense_prefix_layers == 1`; this must be
  generalized.
- Layer construction must support:
  - first `N` layers dense, rest MoE;
  - alternating dense/MoE layers.
- Dense layer MLP size must be set explicitly from the schedule.
- Run names should include a compact schedule segment:
  - `ds1d-sh`
  - `ds1d-nosh-am`
  - `ds2d-nosh-am`
  - `dsalt-sh`
  - `dsalt-nosh-am`

Do not change existing baseline run names retroactively.

## Parameter Check

Before launch, instantiate each variant locally and record:

- active params including embeddings/head;
- active non-embedding params;
- total params;
- number of dense layers;
- number of MoE layers;
- effective active MLP hidden units.

The configs above are fixed. The parameter check is for recording and sanity,
not for redesigning the experiment.

## Launch Settings

Start at 275M.

Recommended first wave:

| Variant | Cx | Purpose |
| --- | ---: | --- |
| `1dense_no_shared_am` | 1 and 4 | Shared expert ablation. |
| `2dense_no_shared_am` | 1 and 4 | Dense prefix replacement. |
| `alternating_no_shared_am` | 1 and 4 | Dense/MoE interleaving replacement. |

Use the same systems settings as baseline:

| Size | Cx | Batch tokens | `global_batch_size_seq` | GPUs | EP | Microbatch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 275M | 1 | 262,144 | 32 | 2 | 1 | 16 |
| 275M | 4 | 524,288 | 64 | 4 | 1 | 16 |
| 275M | 8 | 786,432 | 96 | 4 | 1 | 8 |
| 810M | 1 | 262,144 | 32 | 4 | 1 | 4 |
| 810M | 4 | 524,288 | 64 | 8 | 1 | 4 |

Checkpoint/eval settings:

```text
--ladder-evals
--eval-task-set=fast
--eval-interval=2000
--save-interval=999999999
--ephemeral-save-interval=500
--no-pre-train-checkpoint
```

TODO: Decide how validation losses should influence LR choice and promotion.
For now, validation/eval metrics are logged for later analysis only. LR choice
and promotion decisions use final-window training CE loss.

## LR Transfer Protocol

Use powers-of-two around the baseline fitted LR.

Because dense/shared changes can affect optimization more than expert
granularity, use a three-point transfer probe at the first rung:

### Stage 1: 275M Cx1

For each first-wave variant, run:

```text
0.5x, 1x, 2x
```

around the current baseline 275M Cx1 fitted LR. If the baseline is still about
`2e-3`, use:

```text
1e-3, 2e-3, 4e-3
```

Fit a local quadratic in log LR if bracketed. Otherwise, launch one extension in
the improving direction.

### Stage 2: 275M Cx4

Compute an architecture LR multiplier from Cx1:

```text
m_variant = lr*_variant_Cx1 / lr*_baseline_Cx1
```

Predict:

```text
lr_variant_Cx4 = m_variant * lr_baseline_Cx4
```

Run exactly three LRs:

```text
0.5x, 1x, 2x
```

around the predicted LR so each confirmation rung has a fit-able curve.

### Stage 3: Higher Cx / Scale

Promote promising variants to 275M Cx8. If still competitive, backfill Cx16.

Promote at most one or two variants to 810M Cx1/Cx4 unless this axis becomes the
main project focus.

## Monitoring

Check:

- W&B run appears.
- loss is finite and decreasing;
- `optim/step skipped == 0`;
- throughput is comparable to the baseline family;
- dense/MoE layer counts are logged or inferable from run config;
- checkpoint path matches run name.

Alternating variants may have different throughput. Record actual TFLOPs/GPU and
tokens/sec before comparing wall-clock.

## Analysis

Primary metric:

- final-window training CE `avg250M`.

Also report:

- `avg100M`;
- `avg500M`;
- final step loss;
- throughput;
- active/total params;
- eval metrics as observational context only.

Plot outputs should live under:

```text
src/scripts/train/jacobm_olmoe_ladder/plots/shared_expert_dense_schedule/
```

Recommended plots:

- per-Cx U-plots by variant;
- baseline-vs-variant selected-LR comparison across Cx;
- active-param-adjusted comparison if active params differ materially.

## Promotion Criteria

Promote if:

- variant beats or closely matches baseline while simplifying architecture, e.g.
  no shared expert;
- variant improves with Cx even if Cx1 is close;
- variant gives a clear speed or implementation advantage without loss damage.

Do not promote if:

- it loses clearly at both Cx1 and Cx4;
- LR sensitivity is unusually fragile;
- throughput or memory makes the comparison unattractive.

## Documentation Requirements

After launch:

- append all runs to `RUNS.md`;
- add rationale and LR grids to `ANALYSIS.md`;
- commit code/script changes before relying on launchers.

After completion:

- refresh W&B cache for this run family;
- update final-window summaries;
- regenerate plots in the experiment-specific directory;
- push docs and plots.

## Open TODOs

- Decide exact active/total parameter matching policy for shared/no-shared
  comparisons.
- Decide validation-loss use for LR choice and promotion.
- Decide whether to run active-matched no-shared diagnostic in the first wave or
  only after primary no-shared looks promising.
