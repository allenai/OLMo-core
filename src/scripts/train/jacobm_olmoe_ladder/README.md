# JacobM OLMoE Ladder Directory

Start here for the OLMo4 MoE architecture-search ladder.

## Current Source Of Truth

Read in this order:

1. `PROJECT_OVERVIEW.md` - project goal, architecture-search process, and
   decision policy.
2. `HANDOFF.md` - current resume snapshot and active/queued jobs.
3. `CURRENT_PLAN.md` - current operating plan and autonomy bounds.
4. `SETTINGS_AUDIT.md` - canonical optimizer-batch and systems settings.
5. `LR_FITS.md` - LR fitting policy, transfer rule, and current LR centers.
6. `LAUNCH_QUEUE.md` - draft/approved queue bundles before launching more jobs.
7. `PLOTTED_RESULTS.md` - generated completed-run results from cached W&B
   histories.

## Historical / Append-Only Context

- `RUNS.md` is the experiment ledger. It intentionally preserves old names,
  paths, settings, Beaker IDs, and mistakes.
- `ANALYSIS.md` is dated analysis history. It is useful context, but newer
  source-of-truth docs override it when they conflict.
- `HANDOFF_2026-06-06.md` is an old handoff snapshot.
- `ABLATION_AND_RESEARCH_PLAN.md` and `experiment_plans/` contain detailed
  planning material. Some sections predate the current `480m` naming and Cx2
  `b384k` policy; check `SETTINGS_AUDIT.md` before launching from them.

## Current Entry Points

- New training launches should use
  `src/scripts/train/jacobm_olmoe_ladder/moe_a0_ladder.py`.
- `tiny_275m.py` is retained as a historical copy for reproducing old records.
- New launch scripts should use `--model-size=480m`; `mid_480m` remains accepted
  as a compatibility alias.

## Current Operating Mode

Do not run an autonomous tight polling loop by default. Work from explicit queue
bundles, then poll Beaker/W&B when Jacob asks or when jobs are near natural
milestones.

Run names should be semantic and resume-stable. Do not encode nodes, GPU count,
EP, microbatch, cluster, or other systems-only settings in new experiment names;
record those in W&B/Beaker tags and config instead.
