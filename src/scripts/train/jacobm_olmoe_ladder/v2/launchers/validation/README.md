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
