# Validation backfill launchers

Validation runs separately from training and targets only permanent final
checkpoints. Each task loads model weights in eval-only mode, skips optimizer
construction/state, runs the full downstream suite plus LM validation, and
records results in a distinct W&B run.

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

Collect every registered W&B validation summary into a compact coverage page
and a complete JSON metric export with:

```bash
uv run --with wandb python \
  src/scripts/train/jacobm_olmoe_ladder/v2/collect_validation_results.py
```
