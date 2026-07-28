# OLMoE Ladder v2

This directory will hold the post-migration experiment workflow. The wide v1
integration model is the sole comparison baseline for new pretraining,
midtraining, and long-context experiments.

The v2 operating entry points are now:

- [`EXPERIMENT_RULES.md`](EXPERIMENT_RULES.md): master experiment-running,
  tracking, LR-selection, and plotting contract.
- [`NEXT_EXPERIMENTS.md`](NEXT_EXPERIMENTS.md): current ordered architecture
  experiment queue and promotion path.
- [`ADAPTATION_RECIPE_WIP.md`](ADAPTATION_RECIPE_WIP.md): provisional
  dense-ladder-aligned MT/LCE learning-rate, token-budget, and evaluation
  policy, including unresolved items that block implementation.
- [`MXFP8_LADDER.md`](MXFP8_LADDER.md): aggressive 275M MXFP8 LR-sweep plan,
  exact architecture/precision deltas, resource layout, and promotion gates.
- [`PRETRAINING_LAUNCH_SETTINGS.md`](PRETRAINING_LAUNCH_SETTINGS.md): collected
  larger-size Cx1/Cx2 settings and the pending B300 microbatch study.
- [`GEOMETRY_MATCHED_SCALE.md`](GEOMETRY_MATCHED_SCALE.md): exact
  275M/480M/810M/1.2B geometry-matched `expand_v=2` configs and parameter
  audits.
- [`models/`](models/): audited model-config builders and parameter comparisons.
- [`launchers/pretraining/`](launchers/pretraining/): config-driven DDP
  pretraining sweep launcher and intervention manifests.
- [`launchers/midtraining/`](launchers/midtraining/): config-driven,
  weight-only midtraining continuations from final pretraining checkpoints.
- [`launchers/validation/`](launchers/validation/): final-checkpoint eval-only
  launcher and explicit backfill manifests.
- [`RUNS.md`](RUNS.md): live post-migration run ledger.

The launchers are dry-run by default and require explicit `--submit` plus an
experiment name to create external work. The first-hybrid Cx8 midtraining path
is implemented; long-context launchers remain future work. The result and
callback contracts below already apply to every stage.

## Pretraining plots

`plot_pretraining_wave.py` is the v2 training-loss plotting entry point. Each
wave explicitly registers its W&B run IDs and compares one intervention
against the matching-size wide v1 integration baseline plus any explicitly
named architecture controls. Registered waves cover the first GDN hybrid
(`expand_v=1`), geometry-matched `expand_v=2`, geometry plus NoPE, and geometry
plus NoPE with elementwise attention gating, along with the gated-RoPE
interaction control.

```bash
uv run --isolated --with 'wandb==0.21.4' --with matplotlib python \
  src/scripts/train/jacobm_olmoe_ladder/v2/plot_pretraining_wave.py \
  --wave hybrid_gdn_ev1 --refresh-stale-cache

uv run --isolated --with 'wandb==0.21.4' --with matplotlib python \
  src/scripts/train/jacobm_olmoe_ladder/v2/plot_pretraining_wave.py \
  --wave geometry_gdn_ev2_nope_gated --refresh-stale-cache

uv run --isolated --with 'wandb==0.21.4' --with matplotlib python \
  src/scripts/train/jacobm_olmoe_ladder/v2/plot_pretraining_wave.py \
  --wave geometry_gdn_ev2_rope_gated --refresh-stale-cache

uv run --isolated --with 'wandb==0.21.4' --with matplotlib python \
  src/scripts/train/jacobm_olmoe_ladder/v2/plot_pretraining_wave.py \
  --wave geometry_gdn2_ev2_nope_gated --refresh-stale-cache

PYTHONPATH=src uv run --with matplotlib python \
  src/scripts/train/jacobm_olmoe_ladder/v2/plot_c4_validation_scale.py

uv run --isolated --with 'wandb==0.21.4' --with matplotlib python \
  src/scripts/train/jacobm_olmoe_ladder/v2/plot_canonical_gdn2_kda.py \
  --refresh-stale-cache

uv run --isolated --with 'wandb==0.21.4' --with matplotlib python \
  src/scripts/train/jacobm_olmoe_ladder/v2/plot_kda_mxfp8.py \
  --refresh-stale-cache
```

`plot_canonical_gdn2_kda.py` produces separate canonical GDN2 and KDA U-plots
without reference points, then one shared observed-best plot containing only
bracketed curves for wide integration, matching gated-NoPE GDN1 geometry,
original `expand_v=2` GDN2, canonical `expand_v=1`/nonnegative GDN2, and
canonical KDA. It resolves the 32 paired 275M sweep jobs plus the 12 canonical
GDN2 scale-transfer jobs, four canonical-KDA 480M jobs, and 12 separately
named KDA `expand_v=2`/negative-eigenvalue transfers by exact W&B display name.
The latter family gets its own fixed-LR plot and result ledgers rather than
entering the canonical U-plots. Queued jobs can initialize without a registry
edit; duplicate exact names fail closed and require an explicit resume-segment
decision. Pass `--resolve-only` while training is active to audit registration
without publishing partial plots.

`plot_kda_mxfp8.py` is a separate exact-name registry for the aggressive
MXFP8 KDA sweep. It writes an intervention-only U-plot and a best-of plot that
compares only aggressive MXFP8 against BF16 KDA with our matching
`expand_v=2`, negative-eigenvalue settings; canonical KDA is excluded. The
best-of plot spans all four model sizes and retains every finished BF16 KDA
transferred-LR scale point, while a 275M MXFP8 Cx appears only after its curve
satisfies the normal bracketed quadratic-fit rule. Larger MXFP8 panels remain
empty until corresponding runs finish. The result export includes the 275M
LR sweep and registered 480M fixed-transfer cells. Duplicate W&B names fail
closed; verified restart segments must be listed as explicit predecessor/current
chains so the strict final-window collector can combine them deterministically.

The script writes each selected wave into one matching artifact directory and
uses the final-250M-token mean training CE. The 275M outputs follow the strict
v1 LR-selection contract: an intervention-only U-plot and an observed-best
summary against wide. A Cx enters that summary only when its finished points
support a valid quadratic fit with the observed best strictly inside the swept
LR range; the label is still the actual observed-best LR, never the fitted
prediction. If the available points already bracket a valid curve but one or
more planned runs are unfinished, the observed point is shown with a hollow
orange diamond and a dagger on its LR label. This is a provisional-data marker,
not a substitution of the predicted minimum. The separate
`gdn2_fixed_lr_scale_comparison.png` scale-transfer plot adds
480M/810M/1.2B cells as their registered runs finish, but explicitly labels
them fixed-LR comparisons because their LRs were transferred from wide rather
than optimized for hybrid. Missing or live canonical cells remain visibly
pending. `scale_results.json` and `scale_results.md` record all four sizes;
the original result pair remains the detailed 275M comparison. W&B histories
reuse the migrated v1 cache; `--include-running` remains available for
diagnostic tables but running points never enter formal selection.

`plot_c4_validation_scale.py` renders the C4 validation-CE analogue of the
gated-RoPE fixed-LR scale plot. It preserves the exact checkpoint selection
made by the training-loss plot (observed-best at 275M and transferred wide LR
at larger sizes) and leaves running, missing, or unregistered evaluations
blank rather than substituting partial metrics.

Use the analysis-only W&B `0.21.4` pin shown above. W&B `0.28` materializes a
run's full history before applying the requested tail range, which makes a
finished-only refresh unnecessarily slow for the 100K--200K-step scale runs.

The canonical window is fixed at 250M tokens. Regeneration verifies that each
W&B history actually spans that complete interval and fails loudly rather than
publishing a partial tail. A same-run checkpoint replay is detected from the
token counter, de-duplicated by token position with the latest observation
retained, and surfaced in the result table. If a reset creates a separate final
run segment shorter than 250M tokens, list the earlier W&B IDs in that point's
`predecessor_run_ids` and combine them before regenerating the formal artifacts.

## Result Contract

- Pretraining: load training losses and post-training validation results for
  the new run and the wide integration baseline.
- Midtraining: load post-training validation results, but do not build a
  training-loss comparison.
- Long context: load post-training validation and external RULER results, but
  do not build a training-loss comparison.
- Separate v2 validation runs are collected with
  `collect_validation_results.py`; its Markdown file tracks coverage and its
  JSON file retains every exported `eval/*` metric.
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
3. `WandBCallback` and `ConfigSaverCallback` for durable metrics and exact run
   reconstruction.

Evaluator callbacks must be disabled in training jobs, including on finish.
Run validation in separate eval-only jobs from final checkpoints. This avoids
the evaluator memory state that caused illegal-memory failures and prevents
full validation epochs from distorting training throughput and ETA.

## v1 Audit Finding

The existing ladder trainers only attach evaluator callbacks when
`--ladder-evals` or checkpoint-eval mode is enabled. Do not use
`--ladder-evals` for new training. The standalone trainers retain disabled
evaluator definitions for compatibility with eval-only use, but production
training manifests set their evaluator enable flags to false.
