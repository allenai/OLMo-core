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

## Public Architecture Anchors

Recent non-hybrid local/global attention schedules motivate testing sparser
full-attention ratios than our current alternating pattern:

| Model family | Reported schedule | Notes |
| --- | --- | --- |
| Current MoE A0 | 1:1 SWA/full plus final full | Implemented as `pattern=[WINDOW_SIZE, -1]`, `force_full_attention_on_last_layer=True`. |
| GPT-OSS / GPT-3-style sparse attention | Alternating local/full | Public GPT-OSS description says alternating dense and locally banded sparse attention. |
| Gemma 3 | 5 local : 1 global | Google reports repeating 5 local attention layers and 1 global attention layer. |
| MAI-Thinking-1 | 5 local : 1 global | Microsoft report says it follows Gemma 3 periodic attention with 5 local layers and 1 global layer. |
| Qwen3-Next | 3 cheap layers : 1 attention layer | Hybrid Gated DeltaNet/attention, useful as ratio inspiration but not a pure SWA/full control. |

Qwen3 standard MoE does not provide the same clear public local/global SWA
ratio; Hugging Face's Qwen3 config documentation exposes SWA options but
defaults `use_sliding_window` to false. Treat Qwen3-Next as a hybrid-attention
anchor, not a standard SWA/full-attention datapoint.

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
| `attn_1to1_current` | Current 1 SWA : 1 full pattern | Baseline/control. |
| `attn_3to1` | 3 SWA : 1 full | Intermediate efficiency point; close to several hybrid ratios. |
| `attn_5to1` | 5 SWA : 1 full | Gemma 3 / MAI-style local/global ratio. |
| `attn_mostly_swa` | Final full only, plus first full if required | Extreme efficiency sentinel. |

The exact layer indices should be generated per model size from simple rules
rather than hand-coded ad hoc lists.

### Proposed Layer Rules

Let `L = n_layers`. Generate repeating SWA/full patterns and force the final
layer full, matching the current script's final-full behavior.

| Variant | Pattern | Final layer | Approx full fraction |
| --- | --- | --- | ---: |
| `attn_1to1_current` | `[SWA, full]` | forced full | ~50% |
| `attn_3to1` | `[SWA, SWA, SWA, full]` | forced full | ~25% |
| `attn_5to1` | `[SWA, SWA, SWA, SWA, SWA, full]` | forced full | ~17% |
| `attn_mostly_swa` | all SWA | forced full | `1 / L` |

The current implementation sets `force_full_attention_on_first_layer=False` and
`force_full_attention_on_last_layer=True`. Keep first-layer behavior unchanged
unless a smoke test shows instability.

### Full-Layer Counts By Size

Approximate counts under the final-full rule:

| Size | Layers | 1:1 full | 3:1 full | 5:1 full | Mostly-SWA full |
| --- | ---: | ---: | ---: | ---: | ---: |
| 275M | 12 | 6 | 3 | 2 | 1 |
| mid_480m | 16 | 8 | 4 | 3 | 1 |
| 810M | 20 | 10 | 5 | 4 | 1 |
| 1.2B | 22 | 11 | 6 | 4 | 1 |

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
--attention-schedule {attn_1to1_current,attn_3to1,attn_5to1,attn_mostly_swa}
```

Implementation notes:

1. Keep the current behavior as `attn_1to1_current`.
2. Add a helper that maps `(attention_schedule, n_layers)` to a list or pattern
   of full-attention layers.
3. Keep SWA window size unchanged.
4. Log the selected full-attention layer indices in W&B config or run notes.
5. Add compact run-name tags:
   - `attn1to1`
   - `attn3to1`
   - `attn5to1`
   - `attn-mswa`

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
| `attn_3to1` | 1 and 4 | Intermediate reduction in full-attention layers. |
| `attn_5to1` | 1 and 4 | MAI/Gemma-style schedule. |
| `attn_mostly_swa` | 1 and 4 | Extreme sentinel for how much full attention matters. |

Do not run a denser-than-current attention schedule in the first wave. The
public trend and our systems goals point toward fewer full-attention layers, not
more. If all sparse variants underperform badly, revisit denser schedules later.

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
- Decide whether `attn_mostly_swa` keeps first layer full or only final layer full.
- Decide whether to test this on the baseline 48E/top4 block or the winning
  expert-granularity block.
