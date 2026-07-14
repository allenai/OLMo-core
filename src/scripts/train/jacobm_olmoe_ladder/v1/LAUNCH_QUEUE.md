# Launch Queue Bundle

Status: launched and monitoring as of 2026-06-13 17:12 UTC. 1.2B Cx2 and 480M expert-granularity are active; extra 810M expert-granularity promotions are still waiting on idle compute.

This file is the explicit queue surface for the next wave of MoE ladder work. It
replaces the previous autonomous-loop style: queue bundles are approved, launched,
then polled on request or at natural milestones.

## Priority 0: Before Launching

Completed for the 2026-06-12 bundle:

- `SETTINGS_AUDIT.md` confirmed canonical `b384k` Cx2 for every model size.
- Current status check completed before launch.
- New launch names are semantic and resume-stable. Node count, GPU count, EP,
  microbatch, and cluster are W&B/Beaker tags/config, not name components.
- Weekend/extra compute is not currently idle, so additional 810M promotions are
  waiting.

## Priority 1: Baseline Gap To Fill Soon

### 1.2B Cx2 `b384k` Baseline

Purpose: complete the main Cx1/Cx2/Cx4/Cx8 rectangle for 1.2B without using the
stale `b512k` Cx2 plan.

Canonical setting:

| Model | Cx | Batch tokens | `global_batch_size_seq` | GPUs | EP | Microbatch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.2B | 2 | 393,216 | 48 | 8 | 1 | 2 |

Launched LR grid on 2026-06-12:

| LR | Rationale |
| ---: | --- |
| `1.5e-4` | Cold side around transferred 1.2B Cx2 estimate. |
| `3e-4` | Likely center based on 1.2B Cx1/Cx4 and repaired Cx2 behavior. |
| `6e-4` | Hot side. |

Launched jobs:

| LR | Beaker experiment | Status |
| ---: | --- | --- |
| `1.5e-4` | https://beaker.org/ex/01KTZ15R9Q87WDSPN6740YQX06 | Running at 15.639B tokens, W&B `dtd8qeiv`. |
| `3e-4` | https://beaker.org/ex/01KTZ163MP8B3VFGGY7YWHSFB6 | First job failed before training with `ModuleNotFoundError: olmo_core`; experiment has a fresh pending retry. |
| `6e-4` | https://beaker.org/ex/01KTZ16ETE4R5XZYHKSB8SXDXM | Running at 14.950B tokens, W&B `54pt8zj7`. |

Runtime uses the committed compatibility script `tiny_275m.py`; the preferred
new script name `moe_a0_ladder.py` can become the default after it is committed.

## Priority 2: Expert Granularity

Expert granularity remains the first architecture axis to push. Serious variants
are coarse 24E/top2 and fine 96E/top8. Extreme 192E/top16 and ultra 384E/top32
are diagnostic curiosities and should not receive more ladder compute unless
Jacob explicitly promotes them.

### Current checks to monitor

- 810M Cx1/Cx4 best-observed checks for coarse/fine.
- 275M Cx2 repaired `b384k` curves for baseline/coarse/fine, if not already
  fully reflected in generated plots.

### Full-ladder policy for serious variants

For the serious expert-granularity variants, run the full Cx1/Cx2/Cx4/Cx8
ladder at all four model sizes (`275m`, `480m`, `810m`, `1p2b`). Use 275M to
tune/verify variant LR multipliers and transfer those predicted LRs to larger
sizes. Do not run larger-size LR brackets by default unless transfer looks broken
or Jacob explicitly asks for extra bracketing.

Current queue state:

| Priority | Run family | Scope | Notes |
| ---: | --- | --- | --- |
| 1 | 480M expert granularity | Launched Cx1/Cx2/Cx4/Cx8 for `eg24e2k` and `eg96e8k` on 2026-06-12. | IDs are in `RUNS.md`; Cx2 uses repaired `b384k`. |
| 2 | 810M expert granularity | Wait for idle compute before launching more than the already-running Cx1/Cx4 checks. | Do not queue extra 810M jobs until Jacob confirms compute is available. |
| 3 | 1.2B expert granularity | Launch predicted-LR Cx1/Cx2/Cx4/Cx8 after smaller-size transfer checks are sane. | Training-loss signal is acceptable at this stage. |

## Priority 3: Total Sparsity If Extra Compute Arrives

This axis is paused because expert granularity is higher priority, not because
it is rejected.

First-wave variants:

- `sp96e4k` / `high_total_96e_top4`
- `sp192e4k` / `huge_total_192e_top4`

Recommended queue if compute is abundant:

| Priority | Scope | LR policy | Notes |
| ---: | --- | --- | --- |
| 1 | 275M Cx1/Cx2/Cx4/Cx8 for `sp96e4k`, `sp192e4k` | Tune/verify with small LR grids where needed. | Relaunch only rows that were canceled or missing; use repaired Cx2. |
| 2 | 480M + 810M Cx1/Cx2/Cx4/Cx8 for approved sparsity variants | Predicted/transferred LRs only. | Full ladder for real settings; no default larger-size brackets. |
| 3 | 1.2B Cx1/Cx2/Cx4/Cx8 for approved sparsity variants | Predicted/transferred LRs only. | Queue after smaller-size transfer looks sane or if abundant compute would otherwise idle. |

Use systems from `SETTINGS_AUDIT.md`; Cx16 is still not part of the default
initial grid.

## Priority 4: Later Axes, Not Weekend Defaults

Do not launch these until expert granularity and total sparsity have clearer
answers or Jacob explicitly redirects:

- Dense prefix / shared expert schedule.
- Width/depth geometry.
- SWA/full-attention ratio.
- Integrated candidate runs combining multiple wins.

## Polling / Result Update Procedure

When Jacob asks to poll:

1. Check Beaker state for active IDs in `CURRENT_PLAN.md` and `RUNS.md`.
2. Check W&B state and final tokens for newly finished runs.
3. Do not refresh cache for already-finished cached runs. Collect only newly
   finished or uncached runs since the last update.
4. Regenerate plots and `PLOTTED_RESULTS.md` after relevant full-run completions.
5. Update `CURRENT_PLAN.md` with active/finished decisions and append `RUNS.md`
   for any launches/stops.
6. Summarize whether the next queue needs adjustment.
