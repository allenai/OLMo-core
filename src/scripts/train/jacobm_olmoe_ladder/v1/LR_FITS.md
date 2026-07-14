# LR Fits And Transfer Policy

Last refreshed: 2026-06-12.

This file is the durable summary of how ladder LRs are selected. Generated
result tables in `PLOTTED_RESULTS.md` remain the source for exact run links and
avg250M values; this file records the decision process and current centers.

## Selection Metric

Use completed full runs only. Exclude smoke tests, eval-only runs, sanity checks,
failed starts, stopped partials, and live running jobs from LR decisions.

Primary metric is final-window training CE loss averaged over the final 250M
tokens (`avg250M`). Check `avg100M` and `avg500M` as sensitivity tests, but do
not optimize against validation or downstream evals yet. Validation/eval metrics
are observational until the project moves into integrated candidate selection.

## Fitting Procedure

1. For each model/Cx/variant family, collect completed points with matched
   optimizer batch and comparable settings.
2. Plot loss against `log10(lr)`, not raw LR.
3. Identify the local basin. Use the three points around the best observed LR;
   use five only when the neighborhood is clean and clearly same-family.
4. Fit a local quadratic in `log10(lr)` and report the implied `fit LR` only if
   the minimum lies inside the observed bracket.
5. Treat the fitted LR as a decision aid. Prefer a robust bracketed basin and
   sensible transfer behavior over tiny final-window differences.
6. For the initial real architecture axes, tune/verify LRs at 275M, then launch
   larger sizes at predicted/transferred LRs only unless transfer looks broken.

## Transfer Rule

The original size-transfer prior was too hot for this MoE ladder. Empirically,
275M to 810M points imply a much cooler model-size scaling, roughly `alpha ~= -1`
using active params including embeddings, or `alpha ~= -0.9` with active
non-embedding params.

For new serious variants:

- Fit or verify 275M Cx1/Cx2/Cx4/Cx8 first.
- Estimate variant LR multipliers relative to the baseline at 275M.
- Apply those multipliers to the baseline predicted/fitted LR for 480M, 810M,
  and 1.2B.
- Do not run larger-size LR brackets by default.

## Current Baseline Centers

| Model | Cx | Best observed LR | Fit LR | Notes |
| --- | ---: | ---: | ---: | --- |
| 275M | 1 | `2e-3` | `2.13e-3` | canonical current-family Cx1. |
| 275M | 2 | `1.8e-3` | `1.78e-3` | repaired `b384k` Cx2. |
| 275M | 4 | `1.5e-3` | `1.46e-3` | bracketed. |
| 275M | 8 | `1.6e-3` | `1.35e-3` | bracketed; fit cooler than best observed. |
| 480M | 1 | `1.2e-3` | `9.58e-4` | bracketed but sparse 3-point sweep. |
| 480M | 2 | `9e-4` | `9.73e-4` | repaired `b384k` Cx2. |
| 480M | 4 | `8e-4` | `8.5e-4` | bracketed. |
| 810M | 1 | `6e-4` | `6.21e-4` | bracketed. |
| 810M | 2 | pending | pending | repaired `b384k` runs in progress. |
| 810M | 4 | `4e-4` | `5.14e-4` | broad/shallow basin around `4e-4` to `8e-4`. |
| 810M | 8 | `4e-4` | `4.67e-4` | bracketed. |
| 1.2B | 1 | `4e-4` | `4.83e-4` | bracketed. |
| 1.2B | 2 | pending | pending | next baseline gap. |
| 1.2B | 4 | `3e-4` | `3.66e-4` | `3e-4` and `6e-4` close; hot side complete. |
| 1.2B | 8 | pending | pending | one-node replacements running; old 4-node `4e-4` is systems-comparison only. |

## Current 275M Expert-Granularity Centers

| Variant | Cx1 fit | Cx2 `b384k` fit | Cx4 fit | Cx8 fit | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| baseline 48E/top4 | `2.03e-3` | `1.78e-3` | `1.56e-3` | `1.36e-3` | baseline control. |
| coarse 24E/top2 | `1.86e-3` | `1.81e-3` | `1.57e-3` | `1.40e-3` | serious coarse variant. |
| fine 96E/top8 | `2.10e-3` | `1.89e-3` | `1.45e-3` | `1.35e-3` | serious fine variant. |

Diagnostic 192E/top16 and 384E/top32 Cx1 fits are not policy-setting; they are
noisy/diagnostic unless Jacob promotes them.

## Refresh Commands

Refresh generated result tables after full-run completions:

```bash
uv run --with wandb python src/scripts/train/jacobm_olmoe_ladder/write_plotted_results_md.py --refresh-stale-cache
```

Inspect a specific family when a run finishes:

```bash
uv run --with wandb python src/scripts/train/jacobm_olmoe_ladder/analyze_wandb_ladder.py \
  --name-regex '<specific-finished-run-family>' \
  --mode final --finished-only --windows-m 100 250 500 --refresh-stale-cache
```
