# Width/Depth Geometry Experiment Plan

This runbook specifies the backbone shape ablation: deeper/narrower versus
shallower/wider models at controlled active and total parameter budgets.

## Status

Planned, not yet implemented or launched.

Run this after the baseline ladder has stable LR centers. Preferably run after
expert granularity, because width/depth changes are easier to interpret once the
MoE block shape is less arbitrary.

## Goal

Test whether the MoE A0 family prefers depth or width at approximately fixed
active and total parameter counts.

Current baseline shape:

- 275M: `d_model=768`, `d_attn=1024`, `n_layers=12`
- 810M: `d_model=1280`, `d_attn=1536`, `n_layers=20`
- 1.2B: `d_model=1536`, `d_attn=2048`, `n_layers=22`

Keep the standard MoE block fixed unless the expert-granularity experiment has
already selected a new default:

- 48 experts
- top-4 routing
- `moe_hidden_size = d_model`
- one shared expert with `shared_mlp_hidden_size = d_model / 2`
- one dense prefix layer
- GQA with `n_kv_heads = n_heads // 2`

## Hypothesis

MoE models may prefer depth because each token only activates a sparse subset of
expert capacity per layer, so more layers provide more opportunities for sparse
composition. But too much depth can hurt optimization and wall-clock. Wider
models may improve per-layer expressivity but reduce routing/composition depth.

## Candidate Shapes

Use clean multiples for candidate widths. `d_model` does not need to be a power
of two, but the preferred shapes should be boring for kernels and sharding:

- `d_model` should be a multiple of 64, preferably 128.
- `d_attn` must be divisible by `head_dim=128`.
- Prefer even `n_heads` so `n_kv_heads = n_heads // 2` is exactly half.
- Avoid odd-head candidates unless explicitly approved.

The table below is the current candidate set. These are not random/yolo choices:
they are the proposed configurations to evaluate, with one remaining
parameter-count sanity pass before launch.

### 275M Candidates

| Variant | `d_model` | `d_attn` | Layers | Heads | KV heads | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline` | 768 | 1024 | 12 | 8 | 4 | Current control. |
| `deep_narrow` | 640 | 768 | 16 | 6 | 3 | More layers, narrower width. |
| `shallow_wide` | 896 | 1280 | 9 | 10 | 5 | Fewer layers, wider width. |

### 810M Candidates

| Variant | `d_model` | `d_attn` | Layers | Heads | KV heads | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline` | 1280 | 1536 | 20 | 12 | 6 | Current control. |
| `deep_narrow` | 1152 | 1280 | 24-26 | 10 | 5 | Cleaner multiple than 1120; final layer count needs param check. |
| `shallow_wide` | 1408 | 1792 | 16 | 14 | 7 | Fewer layers, wider width. |

### 1.2B Candidates

| Variant | `d_model` | `d_attn` | Layers | Heads | KV heads | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline` | 1536 | 2048 | 22 | 16 | 8 | Current control. |
| `deep_narrow` | 1408 | 1792 | 28 | 14 | 7 | More layers, narrower width. |
| `shallow_wide` | 1792 | 2304 | 16-18 | 18 | 9 | Cleaner width/head count than 1664/2176; final layer count needs param check. |

Still to settle before launch:

- Whether 810M `deep_narrow` should use 24, 25, or 26 layers after exact
  parameter counts.
- Whether 1.2B `shallow_wide` should use 16, 17, or 18 layers after exact
  parameter counts.
- Whether KV heads of 9 for 1.2B `shallow_wide` are acceptable. This is still a
  clean even-head GQA shape (`18` query heads, `9` KV heads), but it differs from
  the common powers-of-two-ish baseline. If we want stricter head regularity, use
  `d_attn=2048` with 16 heads instead and accept a smaller attention-width
  change.

## Matching Policy

Primary target:

- active params including embeddings/head within 5% of the baseline for that
  size;
- total params within 5% of baseline for that size;
- active non-embedding params recorded separately.

If a candidate misses tolerance:

1. Adjust `n_layers` first, using the ranges above.
2. Keep the proposed `d_model` fixed unless the param mismatch is severe.
3. Adjust `d_attn` only if attention params dominate the mismatch or the head
   shape is rejected.
4. Keep `head_dim=128` and `d_attn` divisible by 128.
5. Prefer even `n_heads` so `n_kv_heads = n_heads // 2` is exactly half, unless
   an odd-head candidate is explicitly approved.

Derived fields:

```text
n_heads = d_attn // 128
n_kv_heads = n_heads // 2
moe_hidden_size = d_model
shared_mlp_hidden_size = d_model / 2
dense_layer_mlp = 4 * d_model + shared_mlp_hidden_size
```

## Required Code Changes

Add a model-shape variant option to
`src/scripts/train/jacobm_olmoe_ladder/tiny_275m.py`.

Recommended CLI:

```text
--shape-geometry {baseline,deep_narrow,shallow_wide}
```

Recommended implementation:

1. Keep `MODEL_SIZE_SPECS` for baseline.
2. Add per-size shape override table:

```python
SHAPE_GEOMETRY_SPECS = {
    "275m": {
        "baseline": dict(),
        "deep_narrow": dict(d_model=640, d_attn=768, n_layers=16),
        "shallow_wide": dict(d_model=896, d_attn=1280, n_layers=9),
    },
    "810m": {
        "baseline": dict(),
        "deep_narrow": dict(d_model=1152, d_attn=1280, n_layers="<24-26>"),
        "shallow_wide": dict(d_model=1408, d_attn=1792, n_layers=16),
    },
    "1p2b": {
        "baseline": dict(),
        "deep_narrow": dict(d_model=1408, d_attn=1792, n_layers=28),
        "shallow_wide": dict(d_model=1792, d_attn=2304, n_layers="<16-18>"),
    },
    ...
}
```

3. Apply shape overrides before deriving heads, shared hidden size, MoE hidden
   size, and dense MLP size.
4. Add run-name segments:
   - `shape-base`
   - `shape-deep`
   - `shape-wide`

Do not change existing baseline run names retroactively.

## Parameter Solver / Dry Check

Before launching, write or reuse a dry-check script that prints:

- active params including embeddings/head;
- active non-embedding params;
- total params;
- `d_model`, `d_attn`, `n_layers`, heads, KV heads;
- attention params estimate;
- MoE routed total params estimate;
- active params per token estimate.

Update this document with the final accepted configs if the starting candidates
are adjusted.

## Launch Settings

Start at 275M.

First wave:

| Variant | Cx | Purpose |
| --- | ---: | --- |
| `deep_narrow` | 1 and 4 | Depth preference probe. |
| `shallow_wide` | 1 and 4 | Width preference probe. |

Use the same baseline systems settings unless dry-run OOM or throughput suggests
adjustment:

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

Width/depth changes may alter optimization more than expert granularity. Use
powers-of-two around the baseline fitted LR, but be prepared to extend.

### Stage 1: 275M Cx1

For both `deep_narrow` and `shallow_wide`, run:

```text
0.5x, 1x, 2x
```

around the current baseline 275M Cx1 fitted LR. If the baseline is about
`2e-3`, use:

```text
1e-3, 2e-3, 4e-3
```

If the best point is on an edge, launch one powers-of-two extension in the
improving direction before making the Cx4 grid.

### Stage 2: 275M Cx4

Compute the architecture LR multiplier:

```text
m_shape = lr*_shape_Cx1 / lr*_baseline_Cx1
```

Run exactly three Cx4 LRs:

```text
0.5x, 1x, 2x
```

around:

```text
m_shape * lr_baseline_275m_Cx4
```

### Stage 3: Higher Cx / Scale

Promote promising variants to 275M Cx8, then Cx16 if still competitive.

Promote at most one shape variant to 810M Cx1/Cx4 initially, unless the deep and
wide variants are both informative and not clearly worse.

## Monitoring

Check:

- W&B run appears.
- loss finite and decreasing;
- `optim/step skipped == 0`;
- throughput and memory relative to baseline;
- shape config logged correctly;
- checkpoint path matches run name.

Deep variants may run slower per token because of more layers. Record both
tokens/sec and actual TFLOPs/GPU.

## Analysis

Primary metric:

- final-window training CE `avg250M`.

Also report:

- `avg100M`;
- `avg500M`;
- final step loss;
- throughput;
- active/total params;
- wall-clock at matched tokens;
- eval metrics as observational context only.

Plot outputs should live under:

```text
src/scripts/train/jacobm_olmoe_ladder/plots/width_depth_geometry/
```

Recommended plots:

- per-Cx U-plots by shape variant;
- baseline-vs-shape selected-LR comparison across Cx;
- loss vs active params and loss vs wall-clock, if wall-clock differs materially.

## Promotion Criteria

Promote if:

- variant beats baseline at Cx1 and Cx4 after LR adjustment;
- variant is close at Cx1 and better at Cx4, suggesting useful scaling;
- variant offers a clear wall-clock or systems advantage at similar loss.

Do not promote if:

- it is clearly worse at both Cx1 and Cx4;
- it requires fragile LR settings;
- it is too slow or memory-heavy to be a plausible standard model.

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

- Generate exact parameter-count table and adjust candidate shapes before launch.
- Settle 810M `deep_narrow` layer count, currently `24-26`.
- Settle 1.2B `shallow_wide` layer count, currently `16-18`.
- Decide whether 1.2B `shallow_wide` should use `d_attn=2304` / 18 heads / 9 KV
  heads, or a stricter `d_attn=2048` / 16 heads / 8 KV heads shape.
- Decide validation-loss use for LR choice and promotion.
- Decide whether the shape experiment should use the current baseline MoE block
  or the winner of expert granularity.
