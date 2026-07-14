# OLMo4 MoE Architecture Search Overview

This directory is the control plane for Jacob's OLMo4 MoE architecture-search
ladder. The immediate goal is to turn the current MoE A0 family into a strong,
well-understood baseline and then test architecture changes against that baseline
without overfitting to one small run.

The long-term goal is to select a base MoE architecture for OLMo4: likely a
roughly 3B-active, 30-40B-total MoE trained to at least 6T tokens, with later
experiments deciding how far multi-epoch Dolma 3 training can be pushed.

## Current Baseline

The current baseline lineage is MoE A0:

- 48 routed experts per MoE layer.
- Top-4 routing.
- `moe_hidden_size = d_model`.
- One shared expert with `shared_mlp_hidden_size = d_model / 2`.
- One dense prefix layer.
- GQA with `n_kv_heads = n_heads // 2`.
- Mostly sliding-window attention with periodic full attention.

Current model-size rungs are named by active parameter count:

- `275m`
- `480m` (`mid_480m` remains a code alias for old launch records)
- `810m`
- `1p2b`

Current data-scale rungs are Cx1, Cx2, Cx4, and Cx8. Reserve Cx16 for important
or unclear cases; do not treat it as part of the default rectangular grid.

## Decision Policy

For the first architecture axes, use final-window training CE as the primary
selection signal. The standard headline metric is `train/CE loss` averaged over
the final 250M tokens (`avg250M`), with `avg100M` and `avg500M` checked for
robustness.

Validation losses and downstream evals are tracked but observational for now.
They become decision signals later, after we have candidate integrated
architectures and start considering mid-training behavior, long-context
extension, downstream quality, and inference/systems performance.

Optimizer-batch matching is the main comparability requirement. Prefer
compute-efficient systems settings over pure wall-clock efficiency when the two
conflict. GPU count, node count, and microbatch may change when compute
constraints require it, as long as optimizer settings remain comparable. Document
those differences in `SETTINGS_AUDIT.md` and W&B/Beaker tags, and plot families
separately when they may matter.

## LR And Ladder Policy

Use completed full runs for LR-selection decisions. Do not use smoke tests,
eval-only runs, stopped partials, or sanity checks as completed U-plot points.

Default LR procedure for baseline calibration:

1. Start each rung with three factor-of-two-spaced LRs centered on a transferred
   or fitted baseline estimate.
2. If the best completed point is on an edge, launch a bounded extension in the
   improving direction.
3. Once a rung has a bracketed completed curve, report both best observed LR and
   a local quadratic fit over loss vs `log10(lr)`.
4. Ignore fitted optima outside the observed bracket.
5. Use variant-specific LR multipliers only after there is enough evidence that
   the multiplier is stable.

For the initial real architecture axes, especially expert granularity and
model/total sparsity, run the full Cx1/Cx2/Cx4/Cx8 ladder across all four model
sizes (`275m`, `480m`, `810m`, `1p2b`). Tune and verify LRs at 275M first, then
use only predicted/transferred LRs for larger sizes unless Jacob explicitly asks
for extra LR bracketing. Later axes may prune this grid once we have a stronger
intuition for transfer behavior.

Cx2 is canonical at 393,216 tokens / 48 sequences (`b384k`) for every model size
unless Jacob explicitly changes the policy.

## Architecture Experiment Stages

Early axes are expected to be relatively straightforward. We are collecting clean
evidence, not trying to discover the whole design space from scratch.

Current / near-term axes:

- Expert granularity: fixed routed active and total capacity while varying
  expert count, `top_k`, and expert width. Serious variants are 24E/top2 and
  96E/top8; 192E/top16 and 384E/top32 are diagnostic curiosities for now. Run
  the full Cx1/Cx2/Cx4/Cx8 by model-size ladder for the serious variants.
- Model/total sparsity at fixed active compute: vary total expert count at fixed
  `top_k=4` and `moe_hidden_size=d_model`. This is paused only because expert
  granularity is higher priority under current compute limits. Once resumed, run
  the same full ladder for the approved real sparsity variants.
- Dense/shared schedule, width/depth geometry, and attention schedule come later.

After independent axes are understood, define integrated candidates that combine
wins, rerun a small confirmation ladder, and then promote only strong candidates
to larger model sizes and richer evaluation.

## Operating Process

Use explicit approved queues rather than an autonomous tight monitoring loop.

Experiment names are semantic resume identifiers. New names should encode the
model, variant, Cx/data scale, batch policy when relevant, LR, and attempt id,
but not node count, GPU count, EP, microbatch, cluster, or other systems-only
settings. Put systems details in W&B/Beaker tags and config so a systems-only
resume can keep the exact same name and checkpoint path.

Preferred workflow:

1. Write or update a launch queue bundle with the exact runs and priorities.
2. Jacob approves or edits the queue.
3. Launch jobs explicitly from the relevant scripts.
4. Poll Beaker/W&B on request or at natural milestones.
5. When full runs complete, refresh W&B caches narrowly, regenerate plots and
   `PLOTTED_RESULTS.md`, then update `CURRENT_PLAN.md` and `RUNS.md`.
6. Propose the next queue only after summarizing completed results.

A 4-hour monitoring cadence is still acceptable for long-running work, but it is
not the default autonomous mode.

## Source-Of-Truth Files

- `PROJECT_OVERVIEW.md`: this project frame and operating philosophy.
- `CURRENT_PLAN.md`: current queue surface, immediate priorities, and next
  decisions.
- `SETTINGS_AUDIT.md`: canonical and historical settings, especially batch and
  systems comparability.
- `LR_FITS.md`: LR fitting method, transfer rule, and current LR centers.
- `PLOTTED_RESULTS.md`: generated completed-run result table from cached W&B
  histories.
- `RUNS.md`: append-only experiment ledger with Beaker IDs and historical notes.
- `ANALYSIS.md`: dated analysis history; useful context, not the current source
  of truth when it conflicts with newer docs.
