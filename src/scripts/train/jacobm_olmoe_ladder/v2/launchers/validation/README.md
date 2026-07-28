# Validation backfill launchers

Validation runs separately from training and targets only permanent final
checkpoints. Each task loads model weights in eval-only mode, skips optimizer
construction/state, runs the configured downstream suite plus LM validation,
and records results in a distinct W&B run.

## Current policy

New backfills target only the final checkpoint selected by observed
final-250M-token training loss for each family/model-size/Cx cell. Do not
backfill every LR point, and do not use a fitted-but-unobserved LR. Fixed-LR
scale-transfer cells have only one eligible checkpoint.

Use the `fast` downstream task set plus LM validation. Eval-only jobs use EP1
at every model size, including 1.2B; evaluation parallelism must not inherit
the training job's EP8 setting. The manifest-level defaults for new work are:

```yaml
evaluation:
  task_set: fast
  expert_parallel_size: 1
  expert_parallel_path: rowwise_nvshmem
```

The path is inactive under EP1, but records the current codebase default rather
than the legacy `sync_1d` fallback. On the completed full-suite backfills, 810M
EP1 took about 67 minutes, while 1.2B EP8 took about 6.5 hours. The 1.2B job
spent 6,733 seconds on the same 578 LM batches for which 810M EP1 needed 199
seconds, so EP8 is not an acceptable eval default. Smoke one 1.2B EP1 winner
before releasing a large backfill batch.

Existing manifests ending in `_full.yaml` are historical records of completed
full-suite evaluations. Do not rewrite them to the new policy.

Inspect without launching:

```bash
uv run --no-sync python \
  src/scripts/train/jacobm_olmoe_ladder/v2/launchers/validation/launch_backfills.py \
  src/scripts/train/jacobm_olmoe_ladder/v2/launchers/validation/manifests/275m_hybrid_geometry_full.yaml
```

Add `--submit --experiment-name NAME` to create the Beaker experiment. Use
`--source-run RUN` (repeatable) for a targeted retry or newly completed cell.

The 2026-07-18 audit found ten missing geometry validations: seven jobs were
externally SIGTERM'd and three never started. The healthy logs contained no
Python/model error, so the retry preserves all evaluation settings and targets
only those ten checkpoints:

```bash
uv run --no-sync python \
  src/scripts/train/jacobm_olmoe_ladder/v2/launchers/validation/launch_backfills.py \
  src/scripts/train/jacobm_olmoe_ladder/v2/launchers/validation/manifests/275m_geometry_missing_full.yaml
```

This renders 10 two-GPU tasks and cannot rerun the 23 completed targets.

Targets may override `model_size`, `expert_parallel_size`,
`expert_parallel_path`, and `rank_microbatch_sequences`. This lets the same
eval-only launcher cover the larger hybrid checkpoints while preserving the
simple 275M EP1 defaults.

The historical RoPE-gated family is split into two manifests so small-model validation
does not reserve eight GPUs unnecessarily:

- `manifests/275m_rope_gated_full.yaml`: 16 two-GPU tasks;
- `manifests/rope_gated_scale_completed_full.yaml`: 10 eight-GPU tasks for
  the currently finished 480M, 810M, and 1.2B checkpoints.

The latter intentionally excludes still-running 810M Cx8 and failed/partial
1.2B Cx2. Add them only after permanent final checkpoints exist.

Collect every registered W&B validation summary into a compact coverage page
and a complete JSON metric export with:

```bash
uv run --with wandb python \
  src/scripts/train/jacobm_olmoe_ladder/v2/collect_validation_results.py
```
