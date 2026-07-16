# Midtraining launchers

`launch_midtraining.py` renders manifest-defined weight-only continuations from
permanent final pretraining checkpoints. It is dry-run by default and requires
both `--submit` and an experiment name to create Beaker work.

The first manifest preserves the established 275M base-model recipe: 100B
tokens at 8K, 128 sequences per optimizer step, four GPUs, EP1, MB8, four-way
gradient accumulation, a fresh optimizer, and 2,000 linear warmup steps into a
constant LR. Evaluators remain disabled during training; full validation runs
post hoc from the permanent final checkpoint.

```bash
uv run --no-sync python \
  src/scripts/train/jacobm_olmoe_ladder/v2/launchers/midtraining/launch_midtraining.py \
  src/scripts/train/jacobm_olmoe_ladder/v2/launchers/midtraining/manifests/275m_hybrid_gdn_ev1_cx8.yaml
```
