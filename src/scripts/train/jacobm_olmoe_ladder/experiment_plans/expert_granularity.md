# Expert Granularity Experiment Plan

This runbook specifies the first post-baseline MoE A0 architecture ablation:
vary expert granularity while preserving routed active and routed total expert
capacity. It is intended to be executable by a future Codex session without
additional context.

## Status

Planned, not yet implemented or launched.

Baseline runs are still in progress as of 2026-06-07. Do not launch this
experiment until the current baseline ladder has enough completed evidence to
define baseline LR centers and compare against current A0.

## Goal

Test whether the current 48-expert top-4 MoE block is the right granularity, or
whether the family improves with fewer larger experts or more smaller experts.

The experiment keeps the dense backbone, shared expert, dense prefix, attention
schedule, routed active expert hidden units, and routed total expert hidden units
fixed. Only these MoE granularity fields change:

- `num_experts`
- `top_k`
- `moe_hidden_size`

## Hypothesis

Fine-grained experts may improve specialization and routing diversity at fixed
active compute, following the intuition behind released fine-grained MoEs. Coarse
experts may be more stable or expressive per expert, following the intuition
behind top-2 coarser MoEs. We need data because the best point may depend on
model size and data scale.

## Variants

The clean invariant family is:

```text
num_experts / top_k = 12
top_k * moe_hidden_size = 4 * d_model
num_experts * moe_hidden_size = 48 * d_model
```

This matches routed active hidden units and routed total expert hidden units
exactly. Router parameter counts differ slightly because router output dimension
changes with expert count; record this as a known minor mismatch.

### Variant Table

| Variant | Experts | `top_k` | `moe_hidden_size` rule | Interpretation |
| --- | ---: | ---: | ---: | --- |
| `coarse_24e_top2` | 24 | 2 | `2 * d_model` | Coarser Phi-style endpoint. |
| `baseline_48e_top4` | 48 | 4 | `d_model` | Current MoE A0 baseline. |
| `fine_96e_top8` | 96 | 8 | `d_model / 2` | Finer DeepSeek/Qwen-style endpoint. |
| `extreme_192e_top16` | 192 | 16 | `d_model / 4` | Very fine-grained diagnostic probe. |
| `ultra_384e_top32` | 384 | 32 | `d_model / 8` | Extreme granularity stress test. |

Do not add the 144E/top12 variant in the first official wave. It is useful as a
future exploratory sentinel, but the 24/48/96 triangle is cleaner and scales
across all three model sizes. The 192E/top16 and 384E/top32 variants are a later
diagnostic extension after the 96E/top8 variant looked strongest at Cx1/Cx4;
run them as small 275M probes first, not as automatic full-ladder variants.

## Exact Configs

Keep these fields fixed within each size:

- `d_model`
- `d_attn`
- `n_layers`
- `head_dim=128`
- `n_heads=d_attn // head_dim`
- `n_kv_heads=n_heads // 2`
- `num_shared_experts=1`
- `shared_mlp_hidden_size=d_model / 2`
- `dense_prefix_layers=1`
- `dense_layer_mlp=4 * d_model + shared_mlp_hidden_size`

### 275M

| Variant | `d_model` | `d_attn` | Layers | Heads | KV heads | Experts | `top_k` | `moe_hidden_size` | Shared hidden | Dense MLP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `coarse_24e_top2` | 768 | 1024 | 12 | 8 | 4 | 24 | 2 | 1536 | 384 | 3456 |
| `baseline_48e_top4` | 768 | 1024 | 12 | 8 | 4 | 48 | 4 | 768 | 384 | 3456 |
| `fine_96e_top8` | 768 | 1024 | 12 | 8 | 4 | 96 | 8 | 384 | 384 | 3456 |
| `extreme_192e_top16` | 768 | 1024 | 12 | 8 | 4 | 192 | 16 | 192 | 384 | 3456 |
| `ultra_384e_top32` | 768 | 1024 | 12 | 8 | 4 | 384 | 32 | 96 | 384 | 3456 |

### 810M

| Variant | `d_model` | `d_attn` | Layers | Heads | KV heads | Experts | `top_k` | `moe_hidden_size` | Shared hidden | Dense MLP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `coarse_24e_top2` | 1280 | 1536 | 20 | 12 | 6 | 24 | 2 | 2560 | 640 | 5760 |
| `baseline_48e_top4` | 1280 | 1536 | 20 | 12 | 6 | 48 | 4 | 1280 | 640 | 5760 |
| `fine_96e_top8` | 1280 | 1536 | 20 | 12 | 6 | 96 | 8 | 640 | 640 | 5760 |
| `extreme_192e_top16` | 1280 | 1536 | 20 | 12 | 6 | 192 | 16 | 320 | 640 | 5760 |
| `ultra_384e_top32` | 1280 | 1536 | 20 | 12 | 6 | 384 | 32 | 160 | 640 | 5760 |

### 1.2B

| Variant | `d_model` | `d_attn` | Layers | Heads | KV heads | Experts | `top_k` | `moe_hidden_size` | Shared hidden | Dense MLP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `coarse_24e_top2` | 1536 | 2048 | 22 | 16 | 8 | 24 | 2 | 3072 | 768 | 6912 |
| `baseline_48e_top4` | 1536 | 2048 | 22 | 16 | 8 | 48 | 4 | 1536 | 768 | 6912 |
| `fine_96e_top8` | 1536 | 2048 | 22 | 16 | 8 | 96 | 8 | 768 | 768 | 6912 |
| `extreme_192e_top16` | 1536 | 2048 | 22 | 16 | 8 | 192 | 16 | 384 | 768 | 6912 |
| `ultra_384e_top32` | 1536 | 2048 | 22 | 16 | 8 | 384 | 32 | 192 | 768 | 6912 |

## Required Code Changes

Add an explicit architecture variant option to
`src/scripts/train/jacobm_olmoe_ladder/tiny_275m.py`.

Recommended approach:

1. Keep `MODEL_SIZE_SPECS` as the size/backbone table.
2. Add an expert-geometry variant table:

```python
EXPERT_GEOMETRY_SPECS = {
    "baseline_48e_top4": dict(num_experts=48, top_k=4, moe_hidden_mult=1.0),
    "coarse_24e_top2": dict(num_experts=24, top_k=2, moe_hidden_mult=2.0),
    "fine_96e_top8": dict(num_experts=96, top_k=8, moe_hidden_mult=0.5),
    "extreme_192e_top16": dict(num_experts=192, top_k=16, moe_hidden_mult=0.25),
    "ultra_384e_top32": dict(num_experts=384, top_k=32, moe_hidden_mult=0.125),
}
```

3. Add CLI option:

```text
--expert-geometry {baseline_48e_top4,coarse_24e_top2,fine_96e_top8,extreme_192e_top16,ultra_384e_top32}
```

4. In `configure_model_size`, apply the selected expert-geometry variant after
   loading the size spec:

```python
geom = EXPERT_GEOMETRY_SPECS[opts.expert_geometry]
NUM_EXPERTS = geom["num_experts"]
TOP_K = geom["top_k"]
MOE_HIDDEN_SIZE = int(spec.d_model * geom["moe_hidden_mult"])
```

5. Keep shared expert and dense prefix unchanged from the model-size spec.
6. Add the expert-geometry name to the run name/tag. Suggested name segment:

```text
eg24e2k
eg48e4k
eg96e8k
eg192e16k
eg384e32k
```

Do not change the existing baseline run names retroactively. New runs should
encode the expert geometry explicitly.

## Parameter Check

Before launching training, run a local parameter-count dry check for every
candidate variant at 275M and 810M. Record:

- active params including embeddings/head;
- active non-embedding params;
- total params;
- routed active hidden units: `top_k * moe_hidden_size`;
- routed total hidden units: `num_experts * moe_hidden_size`.

Add the table to `ABLATION_AND_RESEARCH_PLAN.md` or this file once generated.

### 275M Dry-Run Counts

Verified locally on 2026-06-11 with `--dry-run`.

| Variant | Active params incl emb/head | Active non-embedding params | Total params | Routed active hidden | Routed total hidden |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline_48e_top4` | ~0.28B | ~0.20B | ~1.13B | 3,072 | 36,864 |
| `coarse_24e_top2` | ~0.28B | ~0.20B | ~1.13B | 3,072 | 36,864 |
| `fine_96e_top8` | ~0.28B | ~0.20B | ~1.14B | 3,072 | 36,864 |
| `extreme_192e_top16` | ~0.28B | ~0.20B | ~1.14B | 3,072 | 36,864 |
| `ultra_384e_top32` | ~0.28B | ~0.20B | ~1.14B | 3,072 | 36,864 |

The finer variants have slightly more total parameters because the router output
dimension scales with expert count. This is expected and is the known minor
mismatch in the otherwise fixed routed-capacity family. The 192E/top16 and
384E/top32 rows were dry-run checked locally on 2026-06-11.

## Launch Settings

Use the same systems settings as the corresponding baseline size/rung unless a
smoke test shows memory trouble.

For this first architecture experiment, intentionally run the full 275M ladder
for both non-baseline variants. This is more than the minimum needed for a
screen, but it builds the operational muscle for future ablations and gives us a
clean baseline-vs-variant comparison across the same Cx range.

The first launched full-ladder wave is:

- `coarse_24e_top2`, 275M, Cx1/Cx2/Cx4/Cx8/Cx16.
- `fine_96e_top8`, 275M, Cx1/Cx2/Cx4/Cx8/Cx16.

Recommended launch order:

1. Implement `--expert-geometry`.
2. Smoke test both non-baseline variants.
3. Launch the 275M Cx1 transfer probes for both variants.
4. Fit each variant's Cx1 LR multiplier.
5. Queue the remaining 275M Cx2/Cx4/Cx8/Cx16 runs for both variants, using
   three LRs per rung centered on the transferred baseline optimum.

If we decide that wall-clock matters more than learning the multiplier first, a
reasonable fallback is to queue the whole 275M ladder immediately using
`m_variant = 1.0` for the initial centers, then extend any unbracketed rungs.

Operational overnight exception: if the Cx1 transfer probes will complete when
no human is available to approve the next rung, queue the 275M Cx4 jobs centered
on the existing 275M baseline Cx4 optimum. Use `8e-4`, `1.6e-3`, and `3.2e-3`
for both non-baseline variants. If Cx1 later shows imperfect LR transfer or Cx4
lands on an edge, extend the Cx4 curve by one targeted run per variant under the
usual bracketing rules.

After Cx1/Cx4 completed, Jacob approved the next 275M wave:

- Cx2 for both non-baseline variants with LRs `5e-4`, `1e-3`, `2e-3`.
- Cx8 for both non-baseline variants with LRs `8e-4`, `1.6e-3`, `3.2e-3`.
- Use 2 GPUs for Cx2; coarse uses `mb16`, fine uses `mb8`.
- Use 8 GPUs for Cx8 with `mb4`; `mb8` does not divide the 96-sequence global
  batch evenly across 8 GPUs.

Use the settings below for future launches unless a smoke test shows memory
trouble or Jacob explicitly changes the resource policy. GPU counts follow the
resource table Jacob shared from Yashas: 275M uses `1/2/4/8` GPUs for
Cx1/Cx2/Cx4/Cx8, 810M uses `8/8/8/16`, and the largest rung uses
`8/8/16/32`. EP stays `1` for all rows. Batch tokens are derived from
`global_batch_size_seq * 8192`.

### Canonical Systems Settings

| Size | Cx | Batch tokens | `global_batch_size_seq` | GPUs | EP | Microbatch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 275M | 1 | 262,144 | 32 | 1 | 1 | 16 |
| 275M | 2 | 524,288 | 64 | 2 | 1 | 16 |
| 275M | 4 | 524,288 | 64 | 4 | 1 | 16 |
| 275M | 8 | 786,432 | 96 | 8 | 1 | 8 |
| 275M | 16 | 1,048,576 | 128 | 8 | 1 | 16 |
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

Notes:

- Existing 1.2B Cx4 baseline runs used the older `gpu8-ep1mb2` setting. Future
  comparable launches should use the updated `gpu16-ep1mb2` resource policy
  unless we intentionally need to match those exact historical runs.
- The 275M fine-grained `96E/top8` Cx1 smoke OOMed at `gpu1-ep1mb16`; use
  `gpu1-ep1mb8` for that variant unless a later smoke proves a higher
  microbatch is healthy. The coarse `24E/top2` variant keeps `gpu1-ep1mb16`.
- Hold off on new 810M/1.2B/midpoint Cx16 jobs until the baseline plan changes.
  The 275M expert-granularity experiment is the exception: run Cx16 for both
  non-baseline variants so the first ablation has a full 275M ladder.

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

Use powers-of-two around the baseline fitted LR. Do not run full dense LR grids
for every variant/rung unless transfer fails.

Terminology:

```text
lr_variant(Cx, size) = architecture_multiplier * lr_baseline(Cx, size)
```

### Stage 1: 275M Cx1 Transfer Probe

For both non-baseline variants:

- `coarse_24e_top2`
- `fine_96e_top8`

Run three LRs around the 275M baseline Cx1 fitted optimum:

```text
0.5x, 1x, 2x
```

This uses the measured 275M baseline Cx1 optimum, not a hand-picked LR list.
Using the current approximate baseline Cx1 optimum `~2e-3`, the initial grid is:

```text
1e-3, 2e-3, 4e-3
```

If the current baseline Cx1 fit changes materially before launch, recompute
these powers-of-two around the updated baseline value.

Fit a local quadratic in log LR if all three points bracket. Otherwise, use the
best point and launch one extension in the improving direction.

Compute:

```text
m_variant = lr*_variant / lr*_baseline
```

### Stage 2: 275M Cx4 Confirmation

Predict:

```text
lr_variant(Cx4) = m_variant * lr_baseline_275m(Cx4)
```

Run exactly three LRs so the confirmation rung has a curve that can be fit:

- `0.5x` predicted LR;
- `1x` predicted LR;
- `2x` predicted LR.

Using the current approximate baseline Cx4 optimum `~1.5e-3`, if
`m_variant ~= 1`, the natural confirmation grid is:

```text
8e-4, 1.6e-3, 3.2e-3
```

Round to powers-of-two style values, not arbitrary linear spacing.

### Stage 3: 275M Higher-Cx Promotion

For this first experiment, do not treat Cx8/Cx16 as optional promotion rungs.
Run the whole 275M ladder for both variants once the Cx1 transfer probe has
given us a usable architecture LR multiplier.

For each higher-Cx rung:

```text
lr_variant(Cx) = m_variant * lr_baseline_275m(Cx)
```

Launch three powers-of-two-spaced LRs:

- `0.5x` predicted LR;
- `1x` predicted LR;
- `2x` predicted LR.

If the best completed point is on an edge, launch a bounded extension in the
improving direction. If the curve is monotonically improving at the hot edge,
include a farther sentinel rather than only walking by repeated 2x steps.

Do not claim a new standard architecture from only Cx1/Cx4. The full 275M
Cx1/Cx2/Cx4/Cx8/Cx16 ladder is the first comparison surface for this ablation.

### Stage 4: 810M Transfer Check

Run both core non-baseline variants at 810M Cx1 and Cx4 unless one is clearly
broken at 275M.

Reason:

The point of this ablation is to see whether the granularity trend transfers
with scale. 275M alone may mislead because the fine variant has smaller experts
there, while 810M gives the fine variant more usable per-expert width.

Use the 275M architecture multiplier and current 810M baseline fitted LR:

```text
lr_variant_810m(Cx) = m_variant_275m * lr_baseline_810m(Cx)
```

Start with two or three powers-of-two LRs around the prediction. Use full
four-point sweeps only if the prediction is clearly off or the best point is on
an edge.

### Stage 5: 1.2B Promotion

Run 1.2B only for the best or most informative non-baseline variant after 810M
evidence lands.

Candidate 1.2B rungs:

- Cx1 first.
- Cx4 if Cx1 and lower-scale evidence justify it.
- Potentially all Chinchilla multiples that the 1.2B baseline eventually covers.
  If the baseline is trained through Cx16, the strongest architecture-comparison
  claim would compare the promoted variant over the same Cx range, budget
  permitting.

## Monitoring

For every run, check:

- Beaker job started or queued as expected.
- W&B run appears under `ai2-llm/jacobm-olmoe-ladder`.
- `optim/step skipped == 0`.
- Train loss is finite and decreasing.
- Throughput is in the expected range for the size/rung.
- Checkpoint path matches the run name.

Do not use 512-step smokes, sanity runs, partial runs, or failed runs as final
U-plot points.

## Analysis

Primary metric:

- `train/CE loss`, final-token-window `avg250M`.

Also report:

- `avg100M`
- `avg500M`
- final step loss
- skipped steps
- throughput
- eval metrics as observational context only

For each completed U-curve:

1. Plot loss vs log LR.
2. Mark best observed LR.
3. Fit a local quadratic over three points around the basin when bracketed.
4. Report fitted LR only if it lies inside the completed bracket.
5. Compare against baseline at the same model size and Cx.

Plot expectations:

- One plot per Cx per model size.
- One all-Cx plot per model size.
- Separate variant lines or facet by variant.
- Do not mix expert-geometry variants into the baseline plots without clear
  labels.
- Store plots under an experiment-specific directory, for example:

```text
src/scripts/train/jacobm_olmoe_ladder/plots/expert_granularity/
```

Recommended plot families:

- Baseline-only ladder plots should continue to live in the existing baseline
  plot location.
- Expert-granularity U-plots should show LR curves for each variant at a fixed
  model size and Cx.
- Baseline-vs-variant comparison plots should show final-window training loss
  at each Cx using each variant's selected/fitted LR.
- For early analysis, make comparison plots from training loss only. Validation
  losses remain observational until we explicitly decide how to use them.

## File And Run Organization

Keep experiment-specific artifacts separate from the baseline ladder.

Recommended layout:

```text
src/scripts/train/jacobm_olmoe_ladder/experiments/expert_granularity/
  launch_smoke.sh
  launch_275m_cx1.sh
  launch_275m_cx2.sh
  launch_275m_cx4_baseline_centered.sh
  launch_275m_cx8.sh
  launch_275m_cx16.sh
  reproduce_*.sh
```

Keep docs and plots in the existing planning tree:

```text
src/scripts/train/jacobm_olmoe_ladder/experiment_plans/expert_granularity.md
src/scripts/train/jacobm_olmoe_ladder/plots/expert_granularity/
```

Run naming should be compact. The Beaker/W&B run name should encode only the
pieces needed to identify the scientific condition:

```text
eg-275m-cx1-24e2k-lr2e-3-r1
eg-275m-cx1-96e8k-lr2e-3-r1
```

Use W&B tags and `RUNS.md` for the longer metadata:

- experiment: `exp_expert_granularity`
- variant: `eg24e2k` or `eg96e8k`
- size: `275m`
- Cx: `cx1`, `cx2`, ...
- LR tag
- systems tag: `b256k-gpu1-ep1mb16`, etc.

Do not include stable systems settings in every run name unless they differ from
the canonical table above. This keeps checkpoint paths and W&B group names from
getting too long.
  overlays can be added later after the validation-selection policy is decided.

## Promotion Criteria

Promote from 275M to 810M if either condition holds:

- The variant beats baseline at both Cx1 and Cx4 after LR adjustment.
- The variant is close at one rung and better at another, suggesting a useful
  scaling trend.

Do not promote if:

- It is clearly worse at both Cx1 and Cx4.
- It requires a much more fragile LR.
- It has instability, skipped steps, or routing/throughput problems that make
  the comparison unfair.

Promote from 810M to 1.2B only if the variant improves the trend or is needed to
disambiguate whether granularity transfers with scale.

## Documentation Requirements

After every launch wave:

- Append runs to `RUNS.md` with Beaker IDs.
- Add the launch rationale to `ANALYSIS.md`.
- If scripts/config support changes, commit and push before relying on them for
  future launches.

After every completed run:

- Refresh W&B cache for the specific run family.
- Update final-window summaries.
- Regenerate cached plots.
- Push docs and plots to GitHub.

## Open TODOs

- Decide how validation losses should be used for LR choice, model selection, or
  promotion. Current policy: do not use them for LR selection.
- Decide exact plotting conventions for baseline-vs-experiment comparison once
  the first expert-granularity runs finish.
- Generate exact parameter-count table after implementing the CLI variant.
- Decide whether to smoke 96E/top8 at 275M before full Cx1, because router/alltoall
  behavior may differ from the baseline even with matched active capacity.
- Decide whether to include the baseline rerun under explicit
  `baseline_48e_top4` naming for cleaner plots, or rely on existing baseline
  runs as the anchor.
