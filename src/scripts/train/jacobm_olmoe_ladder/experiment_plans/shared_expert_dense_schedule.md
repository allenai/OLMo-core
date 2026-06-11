# Dense Layers And Shared Expert Experiment Plan

This runbook specifies the V0 dense-layer / shared-expert ablation. The old
version mixed dense schedule and shared-expert removal in the same variants; the
current plan splits them into cleaner sub-experiments so results are easier to
interpret.

## Status

Planned, not yet implemented or launched.

Run independently against the MoE A0 baseline, not on top of the
expert-granularity or total-sparsity winner. The integration run comes later,
after each V0 axis has a candidate winner.

## Goal

Answer two separate questions:

1. How many dense prefix layers does this MoE family want if the shared expert
   stays fixed?
2. Is the shared expert useful when dense prefix depth is held fixed at the
   baseline?

Current baseline:

```text
dense layers = [0]
num_shared_experts = 1
shared_mlp_hidden_size = d_model / 2
moe_hidden_size = d_model
```

## Sub-Experiment A: Dense Count With Shared Expert Fixed

Hold fixed:

- 48 experts;
- top-4 routing;
- `moe_hidden_size = d_model`;
- `num_shared_experts = 1`;
- `shared_mlp_hidden_size = d_model / 2`;
- attention schedule;
- model width/depth.

Vary only dense prefix depth.

| Variant | Dense layers | Shared expert | Routed hidden | Purpose |
| --- | --- | --- | ---: | --- |
| `dense0_shared` | none | `d_model / 2` | `d_model` | Is the dense prefix needed? |
| `dense1_shared` | `[0]` | `d_model / 2` | `d_model` | Current control. |
| `dense2_shared` | `[0, 1]` | `d_model / 2` | `d_model` | Does a larger dense prefix help? |
| `dense4_shared` | `[0, 1, 2, 3]` | `d_model / 2` | `d_model` | Larger dense-prefix sentinel. |

This changes the number of dense versus MoE layers, so active/total parameters
will move. Report those changes explicitly rather than claiming exact parameter
matching.

## Sub-Experiment B: Shared Expert Ablation With Dense Count Fixed

Hold fixed:

- dense layer `[0]`;
- 48 experts;
- top-4 routing;
- attention schedule;
- model width/depth.

Compare shared expert presence.

| Variant | Dense layers | Shared expert | `moe_hidden_size` | Purpose |
| --- | --- | --- | ---: | --- |
| `dense1_shared` | `[0]` | `d_model / 2` | `d_model` | Current control. |
| `dense1_no_shared_am` | `[0]` | none | `9/8 * d_model` | Active-matched quality comparison. |
| `dense1_no_shared_unmatched` | `[0]` | none | `d_model` | Optional cheaper/simpler-model comparison. |

The active-matched no-shared variant uses:

```text
baseline active MLP hidden = 4 * d_model + d_model / 2 = 4.5 * d_model
no-shared active MLP hidden = 4 * moe_hidden_size
moe_hidden_size = 9/8 * d_model
```

The unmatched no-shared variant is not part of the minimal first wave. Add it
only if compute is abundant or if we care about the practical "remove shared
expert and save params/compute" question.

## First-Wave Variants

Run at 275M Cx1/Cx4 first:

| Variant | Sub-experiment | Cx | Role |
| --- | --- | ---: | --- |
| `dense0_shared` | Dense count | 1 and 4 | No-dense-prefix test. |
| `dense2_shared` | Dense count | 1 and 4 | More dense prefix. |
| `dense4_shared` | Dense count | 1 and 4 | Dense-prefix sentinel. |
| `dense1_no_shared_am` | Shared ablation | 1 and 4 | Active-matched no-shared test. |

Optional if compute is abundant:

| Variant | Cx | Role |
| --- | ---: | --- |
| `dense1_no_shared_unmatched` | 1 and 4 | Cheaper no-shared practical comparison. |

Defer alternating dense/MoE schedules. They mix dense schedule and shared-expert
semantics too strongly for the first clean pass. Revisit after the dense-count
and shared-ablation sub-experiments are understood separately.

## Exact Baseline Fields By Size

Use these as fixed backbone fields unless the plan is updated:

| Size | `d_model` | `d_attn` | Layers | Heads | KV heads | Experts | `top_k` | `moe_hidden_size` | Shared hidden | Dense MLP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 275M | 768 | 1024 | 12 | 8 | 4 | 48 | 4 | 768 | 384 | 3456 |
| mid_480m | 1024 | 1024 | 16 | 8 | 4 | 48 | 4 | 1024 | 512 | 4608 |
| 810M | 1280 | 1536 | 20 | 12 | 6 | 48 | 4 | 1280 | 640 | 5760 |
| 1.2B | 1536 | 2048 | 22 | 16 | 8 | 48 | 4 | 1536 | 768 | 6912 |

### Exact 275M First-Wave Configs

| Variant | Dense layers | Experts | `top_k` | `moe_hidden_size` | Shared hidden | Dense MLP |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `dense0_shared` | `[]` | 48 | 4 | 768 | 384 | 3456 |
| `dense1_shared` | `[0]` | 48 | 4 | 768 | 384 | 3456 |
| `dense2_shared` | `[0, 1]` | 48 | 4 | 768 | 384 | 3456 |
| `dense4_shared` | `[0, 1, 2, 3]` | 48 | 4 | 768 | 384 | 3456 |
| `dense1_no_shared_am` | `[0]` | 48 | 4 | 864 | 0 | 3456 |
| `dense1_no_shared_unmatched` | `[0]` | 48 | 4 | 768 | 0 | 3456 |

Derived fields for other sizes:

```text
dense MLP = 4 * d_model + d_model / 2 for shared variants
dense MLP = 4.5 * d_model for active-matched no-shared variants
moe_hidden_size = d_model for shared and no-shared-unmatched variants
moe_hidden_size = 9/8 * d_model for active-matched no-shared variants
```

## Required Code Changes

Add a dense/shared schedule option to
`src/scripts/train/jacobm_olmoe_ladder/tiny_275m.py`.

Recommended CLI:

```text
--dense-schedule {dense1_shared,dense0_shared,dense2_shared,dense4_shared,dense1_no_shared_am,dense1_no_shared_unmatched}
```

Implementation notes:

1. Generalize the current hard-coded dense block override at layer 0.
2. Support arbitrary dense-prefix layer lists for `[]`, `[0]`, `[0, 1]`, and
   `[0, 1, 2, 3]`.
3. Keep alternating dense/MoE schedules out of the first implementation unless
   the plan changes.
4. Set shared expert fields and `moe_hidden_size` from the selected
   dense-schedule variant.
5. Add compact run-name tags:
   - `ds0-sh`
   - `ds1-sh`
   - `ds2-sh`
   - `ds4-sh`
   - `ds1-nosh-am`
   - `ds1-nosh`

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

Use the current canonical systems settings unless a smoke test shows memory
trouble. EP stays `1` for all rows.

| Size | Cx | Batch tokens | `global_batch_size_seq` | GPUs | EP | Microbatch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 275M | 1 | 262,144 | 32 | 1 | 1 | 16 |
| 275M | 2 | 524,288 | 64 | 2 | 1 | 16 |
| 275M | 4 | 524,288 | 64 | 4 | 1 | 16 |
| 275M | 8 | 786,432 | 96 | 8 | 1 | 8 |
| mid_480m | 1 | 262,144 | 32 | 4 | 1 | 8 |
| mid_480m | 2 | 524,288 | 64 | 4 | 1 | 8 |
| mid_480m | 4 | 524,288 | 64 | 4 | 1 | 8 |
| mid_480m | 8 | 786,432 | 96 | 8 | 1 | 4 |
| 810M | 1 | 262,144 | 32 | 8 | 1 | 4 |
| 810M | 2 | 524,288 | 64 | 8 | 1 | 4 |
| 810M | 4 | 524,288 | 64 | 8 | 1 | 4 |
| 810M | 8 | 786,432 | 96 | 16 | 1 | 4 |
| 1.2B | 1 | 262,144 | 32 | 8 | 1 | 2 |
| 1.2B | 2 | 524,288 | 64 | 8 | 1 | 2 |
| 1.2B | 4 | 524,288 | 64 | 16 | 1 | 2 |
| 1.2B | 8 | 786,432 | 96 | 32 | 1 | 1 |

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

### Stage 1: 275M Cx1

For each first-wave variant, run:

```text
0.5x, 1x, 2x
```

around the current baseline 275M Cx1 fitted LR. If the baseline remains about
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

Run exactly three Cx4 LRs around:

```text
m_variant * lr_baseline_275m_Cx4
```

### Stage 3: Higher Cx / Scale

Promote promising variants to 275M Cx8. If still competitive, backfill Cx16.

Promote at most one dense/shared variant to 810M Cx1/Cx4 unless this axis
becomes the main project focus.

## Monitoring

Check:

- W&B run appears;
- loss finite and decreasing;
- skipped steps remain zero;
- throughput and memory relative to baseline;
- dense layer count logged correctly;
- shared expert configuration logged correctly;
- checkpoint path matches run name.

For no-shared variants, inspect whether removing the shared route changes router
load balance or dead/overused experts.

## Analysis

Primary metric:

- final-window training CE `avg250M`.

Also report:

- `avg100M`;
- `avg500M`;
- final step loss;
- throughput;
- active params;
- total params;
- dense layer count;
- MoE layer count;
- shared expert setting;
- eval metrics as observational context only.

Recommended plots:

- dense-count U-plots by Cx;
- shared-ablation U-plots by Cx;
- selected-LR baseline-vs-variant ladder plot;
- loss vs dense layer count;
- throughput vs dense layer count.

Plot outputs should live under:

```text
src/scripts/train/jacobm_olmoe_ladder/plots/shared_expert_dense_schedule/
```

## Promotion Criteria

Promote if:

- the variant beats or clearly matches baseline after LR adjustment;
- the result is robust across Cx1 and Cx4;
- throughput/memory cost is acceptable;
- the result gives an interpretable design decision for the integration run.

Do not promote if:

- it wins only because of an unbracketed LR edge;
- it clearly hurts Cx1 and Cx4;
- throughput/memory cost overwhelms a tiny loss gain;
- it introduces obvious instability or routing pathology.

## Integration Notes

Use the dense/shared winner independently in the integration run together with
the expert-granularity and total-sparsity winners. If the winner comes from
dense-count and the shared ablation also helps, combine them only in the
integration confirmation run, not inside this independent ablation.
