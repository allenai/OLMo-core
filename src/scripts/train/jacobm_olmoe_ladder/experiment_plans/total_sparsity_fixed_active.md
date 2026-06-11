# Total Sparsity At Fixed Active Compute Experiment Plan

This runbook specifies the total-sparsity ablation: change the total routed
expert capacity while keeping active routed compute fixed. It complements
`expert_granularity.md`, which keeps total routed capacity fixed while changing
expert granularity.

## Status

Planned, not yet implemented or launched.

Run after, or alongside, the first expert-granularity wave if extra compute is
available. Prefer using the best current expert geometry as the default MoE
block once it is selected; until then, the clean first version should use the
baseline 48E/top4 geometry family with only expert count changed.

The first wave should focus on *more* total expert capacity, not less. Our
baseline is already relatively high-active compared with recent sparse MoEs, so
the low-total / less-sparse variant is a future diagnostic rather than an
initial run target. Do not launch `low_total_24e_top4` / `sp24e4k` in the first
wave.

## Goal

Test whether adding inactive expert capacity improves pretraining loss at fixed
active compute. This directly asks how sparse the model should be: do more total
parameters help when each token still activates the same amount of expert MLP?

## Hypothesis

More total expert capacity should give the router more specialization options
and may improve loss, especially at higher data scales. Too much total capacity
may hurt through worse routing, thinner per-expert data, load-balancing stress,
memory pressure, or lower throughput.

## Matching Policy

Keep fixed:

- `top_k = 4`
- `moe_hidden_size = d_model`
- active routed hidden units: `top_k * moe_hidden_size = 4 * d_model`
- dense backbone
- shared expert
- dense prefix
- attention schedule

Vary:

- `num_experts`
- therefore total routed expert hidden units:
  `num_experts * moe_hidden_size`

This changes total parameters while keeping active parameters approximately
fixed. Router parameters also change with `num_experts`; record that as a real
part of the design.

## Variants

### Core Family

| Variant | Experts | `top_k` | `moe_hidden_size` | Total routed capacity | Active routed capacity |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline_48e_top4` | 48 | 4 | `d_model` | `48 * d_model` | `4 * d_model` |
| `high_total_96e_top4` | 96 | 4 | `d_model` | `96 * d_model` | `4 * d_model` |
| `huge_total_192e_top4` | 192 | 4 | `d_model` | `192 * d_model` | `4 * d_model` |

Future low-total diagnostic, not in the first wave:

| Variant | Experts | `top_k` | `moe_hidden_size` | Total routed capacity | Active routed capacity |
| --- | ---: | ---: | ---: | ---: | ---: |
| `low_total_24e_top4` | 24 | 4 | `d_model` | `24 * d_model` | `4 * d_model` |

The compact run-name tags are:

| Variant | Tag | First wave? |
| --- | --- | --- |
| `low_total_24e_top4` | `sp24e4k` | No; less sparse than baseline. |
| `baseline_48e_top4` | `sp48e4k` | Control only. |
| `high_total_96e_top4` | `sp96e4k` | Yes. |
| `huge_total_192e_top4` | `sp192e4k` | Yes. |

### Approximate Active Fraction

These estimates use the measured baseline active/total counts and assume the
non-expert static parameter block stays fixed while routed expert total
capacity scales linearly. Exact dry-run counts must replace these before final
launch decisions. Always report active params, total params, and active/total
percentage next to loss and throughput for this experiment.

| Size | Variant | Approx active params | Approx total params | Active / total |
| --- | --- | ---: | ---: | ---: |
| 275M | `low_total_24e_top4` | 0.28B | 0.67B | 42% |
| 275M | `baseline_48e_top4` | 0.28B | 1.13B | 25% |
| 275M | `high_total_96e_top4` | 0.28B | 2.06B | 14% |
| 275M | `huge_total_192e_top4` | 0.28B | 3.92B | 7% |
| mid_480m | `low_total_24e_top4` | 0.48B | 1.43B | 34% |
| mid_480m | `baseline_48e_top4` | 0.48B | 2.56B | 19% |
| mid_480m | `high_total_96e_top4` | 0.48B | 4.83B | 10% |
| mid_480m | `huge_total_192e_top4` | 0.48B | 9.37B | 5% |
| 810M | `low_total_24e_top4` | 0.82B | 2.69B | 30% |
| 810M | `baseline_48e_top4` | 0.82B | 4.93B | 17% |
| 810M | `high_total_96e_top4` | 0.82B | 9.42B | 9% |
| 810M | `huge_total_192e_top4` | 0.82B | 18.39B | 4% |
| 1.2B | `low_total_24e_top4` | 1.22B | 4.19B | 29% |
| 1.2B | `baseline_48e_top4` | 1.22B | 7.76B | 16% |
| 1.2B | `high_total_96e_top4` | 1.22B | 14.90B | 8% |
| 1.2B | `huge_total_192e_top4` | 1.22B | 29.17B | 4% |

### Exact 275M Dry-Run Counts

Verified locally on 2026-06-11 by building the 275M config with
`--total-sparsity`.

| Variant | Active params incl emb/head | Total params | Active / total |
| --- | ---: | ---: | ---: |
| `high_total_96e_top4` / `sp96e4k` | 278,856,192 | 2,069,561,856 | 13.47% |
| `huge_total_192e_top4` / `sp192e4k` | 279,667,200 | 3,938,935,296 | 7.10% |

## Exact Configs

For each model size, keep all baseline fields fixed except `num_experts`.

Baseline size fields:

| Size | `d_model` | `d_attn` | Layers | Heads | KV heads | `top_k` | `moe_hidden_size` | Shared hidden | Dense MLP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 275M | 768 | 1024 | 12 | 8 | 4 | 4 | 768 | 384 | 3456 |
| mid_480m | 1024 | 1024 | 16 | 8 | 4 | 4 | 1024 | 512 | 4608 |
| 810M | 1280 | 1536 | 20 | 12 | 6 | 4 | 1280 | 640 | 5760 |
| 1.2B | 1536 | 2048 | 22 | 16 | 8 | 4 | 1536 | 768 | 6912 |

The `mid_480m` row matches the implemented `MODEL_SIZE_SPECS` in
`tiny_275m.py`.

## Required Code Changes

Add a total-sparsity variant option to
`src/scripts/train/jacobm_olmoe_ladder/tiny_275m.py`.

Recommended CLI:

```text
--total-sparsity {baseline_48e_top4,low_total_24e_top4,high_total_96e_top4,huge_total_192e_top4}
```

Recommended implementation:

1. Keep `MODEL_SIZE_SPECS` as the size/backbone table.
2. Add a total-sparsity spec table that overrides `num_experts` only.
3. Keep `top_k`, `moe_hidden_size`, shared expert, dense prefix, and attention
   unchanged.
4. Add compact run-name tags:
   - `sp24e4k`
   - `sp48e4k`
   - `sp96e4k`
   - `sp192e4k`

Do not change existing baseline run names retroactively.

## Parameter Check

Before launch, instantiate each variant locally and record:

- active params including embeddings/head;
- active non-embedding params;
- total params;
- active/total percentage;
- router parameter count if easy to extract;
- routed active hidden units;
- routed total hidden units.

Replace the approximate table above with exact dry-run counts after
implementation.

## Launch Settings

Start at 275M.

Recommended first wave:

| Variant | Cx | Purpose |
| --- | ---: | --- |
| `high_total_96e_top4` | 1 and 4 | Does 2x total capacity help? |
| `huge_total_192e_top4` | 1 and 4 | Does 4x total capacity help enough to justify cost? |

Do not run `low_total_24e_top4` in the first wave. Save less-sparse variants
for later only if we need to map the full curve or diagnose whether total
capacity is actively hurting.

Use current canonical baseline systems settings as the starting point, but note
that total/optimizer memory grows with expert count even when active compute is
fixed. For the 275M first wave, use 4 GPUs for Cx1/Cx4 and 8 GPUs for Cx8 unless
smokes prove a smaller setting is healthy and worth using.

| Size | Cx1 | Cx2 | Cx4 | Cx8 |
| --- | ---: | ---: | ---: | ---: |
| 275M total-sparsity first wave | 4 GPUs | 4 GPUs | 4 GPUs | 8 GPUs |
| mid_480m | 4 GPUs | 4 GPUs | 4 GPUs | 8 GPUs |
| 810M | 8 GPUs | 8 GPUs | 8 GPUs | 16 GPUs |
| 1.2B | 8 GPUs | 8 GPUs | 16 GPUs | 32 GPUs |

EP stays `1` unless memory forces a smoke-tested fallback. Higher-total variants
may need lower microbatch or more GPUs for memory even though active compute is
fixed; smoke `high_total_96e_top4` and `huge_total_192e_top4` before full runs.

Implemented launchers:

```text
src/scripts/train/jacobm_olmoe_ladder/experiments/total_sparsity/launch_smoke.sh
src/scripts/train/jacobm_olmoe_ladder/experiments/total_sparsity/launch_275m_cx1_cx4.sh
src/scripts/train/jacobm_olmoe_ladder/experiments/total_sparsity/launch_275m_cx8.sh
```

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
Current policy: validation/eval metrics are observational only; LR choice uses
final-window training CE.

## LR Transfer Protocol

Use powers-of-two around the baseline fitted LR.

Because total expert count changes memory/routing behavior but keeps active MLP
capacity fixed, assume LR transfer is plausible but not guaranteed.

Stage 1: 275M Cx1

- Run three LRs per variant: `0.5x`, `1x`, `2x` around the baseline Cx1 fitted
  LR.
- If baseline Cx1 remains near `2e-3`, use `1e-3`, `2e-3`, `4e-3`.
- If the best point is on an edge, launch one factor-of-two extension.

Stage 2: 275M Cx4

- Compute `m_variant = lr*_variant_Cx1 / lr*_baseline_Cx1`.
- Center Cx4 on `m_variant * lr_baseline_275m_Cx4`.
- Run exactly three powers-of-two-spaced LRs.

Stage 3: Higher Cx / Scale

- Promote promising variants to 275M Cx8.
- Promote at most one high-total variant to 810M Cx1/Cx4 initially.
- Promote to 1.2B only if the higher-total variant clearly improves the
  baseline or expert-granularity winner.

## Monitoring

Check:

- W&B run appears.
- loss finite and decreasing;
- skipped steps remain zero;
- throughput and memory relative to baseline;
- router load balance and dead/overused experts;
- checkpoint path matches run name.

For high-total variants, explicitly record wall-clock and memory pressure: the
scientific win is less useful if total capacity slows training dramatically.

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
- active/total percentage;
- eval metrics as observational context only.

Recommended plots:

- per-Cx U-plots by sparsity variant;
- selected-LR baseline-vs-variant comparison across Cx;
- loss vs active/total percentage;
- loss improvement per added total parameter.

Plot outputs should live under:

```text
src/scripts/train/jacobm_olmoe_ladder/plots/total_sparsity_fixed_active/
```

## Promotion Criteria

Promote if:

- higher total capacity beats baseline after LR adjustment;
- the advantage grows with Cx;
- throughput/memory cost is acceptable;
- routing diagnostics do not show severe collapse.

Do not promote if:

- it loses clearly at Cx1 and Cx4;
- it is too LR-sensitive;
- it has severe load imbalance or dead experts;
- the wall-clock cost dominates the loss improvement.

## Documentation Requirements

After launch:

- append runs to `RUNS.md`;
- add rationale and LR grids to `ANALYSIS.md`;
- commit code/script changes before relying on launchers.

After completion:

- refresh W&B cache for this run family;
- update final-window summaries;
- regenerate experiment-specific plots;
- push docs and plots.

## Open TODOs

- Implement `--total-sparsity` or decide to fold this into the existing
  `--expert-geometry` table.
- Replace approximate active/total percentages with exact dry-run counts.
- Smoke high-total variants for memory and throughput.
- Decide whether to run total-sparsity variants on top of baseline 48E/top4 or
  the winning expert-granularity geometry.
