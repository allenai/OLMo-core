# SWA Versus Full Attention Ratio Experiment Plan

This runbook specifies a concrete attention-schedule ablation: change the ratio
of sliding-window attention (SWA) layers to full-attention layers while keeping
the rest of the MoE architecture fixed. This is separate from a larger
hybrid-attention experiment; here we only vary how often the current full
attention path appears.

## Status

Planned, not yet implemented or launched.

Run after expert granularity starts to identify the standard MoE block, or run
in parallel if additional compute is available and the implementation is cheap.

## Goal

Measure how much full attention the MoE A0 family needs for pretraining loss and
observational eval metrics. The current architecture uses a mostly sliding-window
schedule with periodic full attention. We want to know whether fewer full
attention layers preserve loss while improving efficiency, or whether more full
attention layers are worth the cost.

## Hypothesis

Fewer full-attention layers may have similar short-context training CE and better
throughput, but could hurt validation/eval tasks that need broader context.
More full-attention layers may improve quality but cost more wall-clock and
memory. Since LR selection is currently training-loss-only, validation/eval
metrics should be logged but not used for LR choice.

## Variants

Keep fixed:

- model size / backbone shape;
- expert geometry;
- shared expert and dense prefix;
- optimizer and schedule;
- sequence length;
- global batch rule.

Vary only the attention schedule.

Recommended first family:

| Variant | Schedule concept | Interpretation |
| --- | --- | --- |
| `attn_current` | Current SWA/full pattern | Baseline/control. |
| `attn_sparse_full` | Fewer full-attention layers than current | Efficiency-biased. |
| `attn_dense_full` | More full-attention layers than current | Quality-biased. |
| `attn_all_swa` | No periodic full attention except any required boundary layer | Extreme efficiency sentinel. |

The exact layer indices should be generated per model size from simple rules
rather than hand-coded ad hoc lists.

### Proposed Layer Rules

Let `L = n_layers`.

| Variant | Full-attention layers |
| --- | --- |
| `attn_current` | Existing script behavior. |
| `attn_sparse_full` | Layer `0`, final layer, and every 8th layer. |
| `attn_dense_full` | Layer `0`, final layer, and every 2nd layer. |
| `attn_all_swa` | Final layer only, unless the first layer must remain full for stability. |

TODO: Confirm whether the current implementation forces full attention on the
first and/or final layer through `SlidingWindowAttentionConfig` flags. The exact
rule should respect those constraints or make the override explicit.

## Exact Configs

Use the current baseline model sizes unless the expert-granularity winner has
already replaced the default MoE block.

Baseline size fields:

| Size | `d_model` | `d_attn` | Layers | Heads | KV heads | Experts | `top_k` | `moe_hidden_size` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 275M | 768 | 1024 | 12 | 8 | 4 | 48 | 4 | 768 |
| mid_480m | 1024 | 1024 | 16 | 8 | 4 | 48 | 4 | 1024 |
| 810M | 1280 | 1536 | 20 | 12 | 6 | 48 | 4 | 1280 |
| 1.2B | 1536 | 2048 | 22 | 16 | 8 | 48 | 4 | 1536 |

The `mid_480m` row matches the implemented `MODEL_SIZE_SPECS` in
`tiny_275m.py`.

## Required Code Changes

Add an attention-schedule option to
`src/scripts/train/jacobm_olmoe_ladder/tiny_275m.py`.

Recommended CLI:

```text
--attention-schedule {attn_current,attn_sparse_full,attn_dense_full,attn_all_swa}
```

Implementation notes:

1. Keep the current behavior as `attn_current`.
2. Add a helper that maps `(attention_schedule, n_layers)` to a list or pattern
   of full-attention layers.
3. Keep SWA window size unchanged.
4. Log the selected full-attention layer indices in W&B config or run notes.
5. Add compact run-name tags:
   - `attn-cur`
   - `attn-sparse`
   - `attn-dense`
   - `attn-allswa`

Do not change existing baseline run names retroactively.

## Parameter Check

Parameter counts should be identical across attention-schedule variants. Before
launch, dry-run each variant and record:

- active params;
- total params;
- full-attention layer indices;
- SWA layer count;
- full-attention layer count.

The dry check should fail loudly if parameter counts differ unexpectedly.

## Launch Settings

Start at 275M.

Recommended first wave:

| Variant | Cx | Purpose |
| --- | ---: | --- |
| `attn_sparse_full` | 1 and 4 | Can we reduce full attention cheaply? |
| `attn_dense_full` | 1 and 4 | Does more full attention improve quality? |
| `attn_all_swa` | 1 and 4 | Extreme sentinel for how much full attention matters. |

Use current canonical systems settings:

| Size | Cx1 | Cx2 | Cx4 | Cx8 |
| --- | ---: | ---: | ---: | ---: |
| 275M | 1-2 GPUs | 2 GPUs | 4 GPUs | 8 GPUs |
| mid_480m | 4 GPUs | 4 GPUs | 4 GPUs | 8 GPUs |
| 810M | 8 GPUs | 8 GPUs | 8 GPUs | 16 GPUs |
| 1.2B | 8 GPUs | 8 GPUs | 16 GPUs | 32 GPUs |

Checkpoint/eval settings:

```text
--ladder-evals
--eval-task-set=fast
--eval-interval=2000
--save-interval=999999999
--ephemeral-save-interval=500
--no-pre-train-checkpoint
```

TODO: Decide how validation losses should influence promotion. Current policy:
training CE selects LRs; eval metrics are observational.

## LR Transfer Protocol

Attention schedule may affect optimization less than width/depth but more than
pure total-sparsity. Use the standard three-point transfer probe.

Stage 1: 275M Cx1

- Run `0.5x`, `1x`, `2x` around the baseline Cx1 fitted LR.
- If baseline Cx1 is near `2e-3`, use `1e-3`, `2e-3`, `4e-3`.
- Extend once if the best point lands on an edge.

Stage 2: 275M Cx4

- Compute `m_attn = lr*_attn_Cx1 / lr*_baseline_Cx1`.
- Center Cx4 on `m_attn * lr_baseline_275m_Cx4`.
- Run exactly three powers-of-two-spaced LRs.

Stage 3: Higher Cx / Scale

- Promote variants that improve or nearly match loss with better throughput to
  275M Cx8.
- Promote at most one attention-ratio variant to 810M Cx1/Cx4 initially.
- Do not promote solely on short-context train loss if eval metrics show a clear
  context-sensitive regression; flag for human review.

## Monitoring

Check:

- W&B run appears.
- loss finite and decreasing;
- skipped steps remain zero;
- throughput relative to baseline;
- memory relative to baseline;
- full-attention layer indices logged correctly;
- checkpoint path matches run name.

For attention variants, throughput is part of the result. Record tokens/sec,
TFLOPs/GPU, and wall-clock.

## Analysis

Primary metric:

- final-window training CE `avg250M`.

Also report:

- `avg100M`;
- `avg500M`;
- final step loss;
- throughput;
- wall-clock;
- full-attention layer count;
- eval metrics as observational context.

Recommended plots:

- per-Cx U-plots by attention schedule;
- selected-LR baseline-vs-variant comparison across Cx;
- loss vs full-attention layer fraction;
- throughput vs full-attention layer fraction.

Plot outputs should live under:

```text
src/scripts/train/jacobm_olmoe_ladder/plots/swa_full_attention_ratio/
```

## Promotion Criteria

Promote if:

- a lower-full-attention variant matches baseline loss and improves throughput;
- a higher-full-attention variant clearly improves loss enough to justify cost;
- the advantage persists or improves from Cx1 to Cx4.

Do not promote if:

- it loses clearly at both Cx1 and Cx4;
- eval metrics show a large context-sensitive regression;
- throughput or memory cost is unattractive.

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

- Inspect the current attention schedule implementation and record exact
  baseline full-attention layer indices for every model size.
- Decide whether `attn_all_swa` keeps first layer full or only final layer full.
- Decide whether to test this on the baseline 48E/top4 block or the winning
  expert-granularity block.
