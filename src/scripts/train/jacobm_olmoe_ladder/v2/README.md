# OLMoE Ladder v2

This directory will hold the post-migration experiment workflow. The wide v1
integration model is the sole comparison baseline for new pretraining,
midtraining, and long-context experiments.

The v2 operating entry points are now:

- [`EXPERIMENT_RULES.md`](EXPERIMENT_RULES.md): master experiment-running,
  tracking, LR-selection, and plotting contract.
- [`NEXT_EXPERIMENTS.md`](NEXT_EXPERIMENTS.md): current ordered architecture
  experiment queue and promotion path.
- [`PRETRAINING_LAUNCH_SETTINGS.md`](PRETRAINING_LAUNCH_SETTINGS.md): collected
  larger-size Cx1/Cx2 settings and the pending B300 microbatch study.
- [`models/`](models/): audited model-config builders and parameter comparisons.
- [`launchers/pretraining/`](launchers/pretraining/): config-driven DDP
  pretraining sweep launcher and intervention manifests.
- [`RUNS.md`](RUNS.md): live post-migration run ledger.

The launcher is dry-run by default and requires explicit `--submit` plus an
experiment name to create external work. Midtraining and long-context launchers
remain future work; their result and callback contracts below already apply.

## Pretraining plots

`plot_pretraining_wave.py` is the v2 training-loss plotting entry point. Each
wave explicitly registers its W&B run IDs and compares one intervention only
against the wide v1 integration baseline. The first registered wave is the
275M-active GDN hybrid (`expand_v=1`) at Cx1/2/4/8.

```bash
.venv/bin/python \
  src/scripts/train/jacobm_olmoe_ladder/v2/plot_pretraining_wave.py \
  --wave 275m_hybrid_gdn_ev1 --include-running --refresh-stale-cache
```

The script uses the final-250M-token mean training CE and follows the v1 plot
contract: one intervention-only all-Cx U-plot and one observed-best summary that
compares the intervention with the wide baseline. It also writes JSON/Markdown
result tables. A Cx enters the summary only when its finished intervention
points support a valid quadratic fit with the observed best strictly inside the
swept LR range; the summary label is still the actual observed-best LR, never
the fitted prediction. Only finished runs are eligible for plots or selection;
incomplete sweeps remain marked provisional in the result table.
W&B histories reuse the migrated v1 cache, and running jobs fetch and cache only
a tail window.

## Result Contract

- Pretraining: load training losses and in-loop evals for the new run and the
  wide integration baseline.
- Midtraining: load in-loop evals, but do not build a training-loss comparison.
- Long context: load in-loop evals and external RULER results, but do not build a
  training-loss comparison.
- Run external RULER through converted HF checkpoints and vLLM on Jupiter. The
  current `olmo-eval` OLMo-core provider does not load OLMo-DDP checkpoints
  whose state keys use the `module.*.main` layout.
- Do not automatically compare new runs against every v1 intervention.

## Trainer Callback Contract

Every v2 pretraining, midtraining, and long-context trainer must attach:

1. `SpeedMonitorCallback`, which computes the rolling MFU used in training
   progress.
2. `BeakerCallback`, which writes `trainer.training_progress` into the Beaker
   description. `training_progress` already includes the rolling MFU from the
   speed monitor, so a second MFU-specific callback is not needed.
3. In-loop language-model validation through `LMEvaluatorCallbackConfig`.
4. In-loop downstream evaluation through `DownstreamEvaluatorCallbackConfig`.
5. `WandBCallback` and `ConfigSaverCallback` for durable metrics and exact run
   reconstruction.

The scale-hybrid trainer now implements these callbacks. Production manifests
select the v1 `fast` evaluation group every 2,000 steps and on finish; smoke
manifests retain the short HellaSwag-only probe.

## v1 Audit Finding

The existing ladder trainers only attach in-loop eval callbacks when
`--ladder-evals` or checkpoint-eval mode is enabled. They do not explicitly
attach `BeakerCallback`. The standalone 275M long-context trainer still needs
the centralized v2 callback contract; the scale-hybrid trainer has been brought
into compliance.
