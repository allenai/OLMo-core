# MoE A0 Ladder Process

This is the operating guide for the JacobM OLMoE ladder runs. Use it when
starting a new session, splitting monitoring across multiple sessions, or
onboarding someone new.

The live project state is split across three files:

- `RUNS.md`: append-only-ish experiment ledger with Beaker IDs and run notes.
- `ANALYSIS.md`: current interpretation of results, W&B commands, and decisions.
- `LADDER_PROCESS.md`: process rules and experiment-management philosophy.

## Goals

The near-term goal is to tune the MoE A0 architecture at 275M active parameters
through Cx16 before scaling token counts or moving fully to larger model sizes.
The larger-model work is a baseline-transfer check, not a new architecture
search.

Model sizes in this lineage:

- `275m`: current architecture search scale, about 278M active params.
- `810m`: first larger baseline, same MoE structure, EP=1 preferred.
- `1p2b`: prepared and smoke-tested, but do not launch full LR probes without
  explicit approval.

## Run Families

Do not blindly merge results from different systems settings into one U-curve.
We have seen meaningful run-family differences.

Important families so far:

- `original`: early runs before the throughput fixes.
- `n2`: early two-node high-side Cx1 probes.
- `gpu2-ep1mb16`: current canonical Cx1/Cx2 family.
- `gpu4-ep1mb16`: current canonical Cx4 family.
- `gpu4-ep1mb8`: current canonical Cx8 family.
- `gpu8-ep1mb16`: current canonical Cx16 family.

Per-rung plots should show separate families when multiple exist. Aggregate model
plots should show only the canonical family for each rung unless explicitly
debugging run-family effects.

Diagnostic runs with `sanity` in the name are for controlled settings checks and
should be excluded from ladder LR fits. The EP=8 Cx1 sanity run is an example:
it tests whether EP, rather than microbatch size, explains the Cx1 family gap.

## Default Measurement

Use final-token-window training CE, not final-step CE alone.

Default windows:

- `avg100M`
- `avg250M`
- `avg500M`

`avg250M` is the primary headline number. Check `avg100M` and `avg500M` for
robustness, especially when curves are close or noisy.

Use completed full runs for LR-selection decisions. In-flight and failed partials
can guide intuition, but they should not be treated as final U-plot points.

## Initial LR Sweeps

Start each new rung with a coarse LR sweep, not a dense grid.

Default pattern:

- Use factor-of-two spacing when uncertain.
- Use 3-4 LRs for an initial probe when runs are expensive.
- Center the new rung around the trend from smaller rungs when there is enough
  evidence.
- Do not launch dense local grids unless the rung is cheap or unusually
  ambiguous.

For early cheap 275M rungs, it is acceptable to run more points to diagnose
systems or family effects. For larger token budgets or larger model sizes, prefer
coarse bracketing plus one targeted refinement.

## Expanding The Search Range

A rung is bracketed only when the completed U-curve has a plausible low side, an
interior best, and a high side.

If the best completed point is on an edge:

- Launch at least one extension in the improving direction.
- If the curve is monotonically improving at the high edge, do not only walk by
  repeated 2x steps. Also launch an occasional true sentinel farther out, around
  an order of magnitude from the current best/high edge, to find the upturn
  quickly.
- Keep nearer extensions running if they may still be promising; use the
  sentinel to establish the right-side shape faster.

Example policy:

- Cx8 completed through `1.6e-3` and was still improving, so launch nearer
  extensions like `3.2e-3`/`6.4e-3` plus a true sentinel such as `1.6e-2`.
- Cx16 completed through `6e-4`, with `1.2e-3` improving in flight, so launch a
  nearer extension like `2.4e-3` plus a true sentinel such as `6e-3`.

If the sentinel is catastrophically bad early, still prefer letting it finish
unless there is an explicit reason to cancel. The project preference is to get
final full-run numbers for launched non-duplicate runs.

## Refinement And LR Rules

Once a rung has at least three bracketed completed points:

- Report best observed LR.
- Fit a quadratic to loss vs `log10(lr)` using 3 or 5 local points.
- Report the fitted optimum if it lies inside the observed bracket.
- Clamp/ignore fits that point outside the bracket; in that case the rung is not
  actually bracketed and should be extended instead.

Use the fitted optimum as a guide, not an oracle. Small final-window differences
can be noise, and run-family effects can dominate if settings changed.

## Moving To The Next Rung

Move on from a rung when:

- The canonical family has a bracketed completed U-curve, or
- The remaining uncertainty is too small to matter for the next transfer
  decision, and a follow-up would be lower value than starting the next rung.

Do not move on just because a partial looks good. Use completed full runs for
the formal decision.

For the 275M ladder, the goal is to complete Cx1/Cx2/Cx4/Cx8/Cx16. Once Cx8 and
Cx16 have usable brackets, the 275M LR rule is good enough to start the 810M
baseline probes.

## Larger Model Plan

Keep architecture fixed for larger baseline checks:

- 48 experts
- top-4 routing
- `moe_hidden_size = d_model`
- one shared expert
- `shared_mlp_hidden_size = d_model / 2`
- one dense prefix layer
- GQA with `n_kv_heads = n_heads // 2`

810M:

- EP=1 by default.
- Validated Cx1 setting: `gpu4-ep1mb4`.
- Use `gpu8-ep1mb4` for Cx4 to speed up, no extra 8-GPU smoke required.
- After 275M Cx8/Cx16 are usable, launch 810M Cx1 coarse LR probes.
- Initial 810M Cx1 LRs: `1e-4`, `2e-4`, `4e-4`, `8e-4`.
- After 810M Cx1 and the 275M LR-rule evidence are available, launch 810M Cx4
  around the transferred rule using 3-4 factor-of-two-spaced LRs.
- Launch 810M Cx8 only if Cx4 is clean or needed to validate the transferred
  LR-rule slope.

1.2B:

- EP=1 by default.
- Preferred smoke-tested setting: `gpu4-ep1mb4`.
- EP=2 is a fallback only if memory forces it.
- Do not launch full 1.2B LR probes without explicit approval.

## Monitoring Cadence

When actively monitoring:

- Wait a real 20 minutes between monitoring cycles.
- Do not poll Beaker/W&B every few seconds unless debugging immediate startup,
  OOM, or launch validity.
- If interrupted, stop any background `sleep 1200` before doing interactive work.

Useful status pattern:

```bash
beaker experiment get <ids...> --format json | \
  jq -r '.[] | [.id, .name, (.jobs[0].status.started // "not-started"), (.jobs[0].status.finalized // "running"), (.jobs[0].status.exitCode // ""), (.jobs[0].requests.gpuCount // "")] | @tsv'
```

Useful W&B summary pattern:

```bash
uv run --with wandb python src/scripts/train/jacobm_olmoe_ladder/analyze_wandb_ladder.py \
  --mode final \
  --name-regex 'olmoe3-tiny-275m-cx'
```

Regenerate completed-run plots after new full runs finish:

```bash
uv run --with wandb --with matplotlib python src/scripts/train/jacobm_olmoe_ladder/plot_wandb_ladder.py \
  --window-m 250 \
  --finished-only
```

Commit and push docs/plot updates when full-run results change.

## Launch And Cancellation Rules

Allowed without asking, under the active goal:

- Inspect Beaker/W&B.
- Update docs and plots.
- Commit and push docs/plot/bookkeeping updates.
- Launch bounded 275M Cx8/Cx16 follow-ups under the LR-selection rules.
- Launch 810M Cx1/Cx4 coarse probes once the prerequisites are met.
- Stop clearly accidental duplicate jobs.

Ask before:

- Launching full 1.2B probes.
- Changing architecture, tokenizer, data mix, optimizer family, or schedule
  shape.
- Launching beyond Cx16.
- Using more than 8 GPUs for one job.
- Cancelling healthy non-duplicate full runs.
- Pushing code/script changes that are not docs, plots, or bookkeeping.

## Checkpoint And Storage Hygiene

Future runs should use final-only permanent checkpoints plus latest ephemeral
checkpoints. Avoid retaining every intermediate checkpoint.

Storage cleanup can be aggressive for stopped, failed, duplicate, or clearly
superseded runs, but be careful with active/resumable jobs. If a run may need to
resume, do not delete its latest ephemeral checkpoint.

## Handoff Checklist

Before handing the project to another session/person:

- List active Beaker IDs and which rung/LR each belongs to.
- Note which active jobs are diagnostic-only.
- State which completed rung is bracketed or still high-edge/low-edge best.
- Update `RUNS.md` with any launched/stopped/ignored jobs.
- Update `ANALYSIS.md` with any new full-run results and current interpretation.
- Regenerate and push plots if full-run results changed.
- Mention any unrelated dirty worktree files so they are not accidentally
  reverted.

