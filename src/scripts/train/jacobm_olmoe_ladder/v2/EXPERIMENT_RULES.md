# OLMoE ladder v2 experiment rules

This is the master operating contract for post-migration architecture
experiments. If a launcher, plot, or informal note conflicts with this file,
stop and resolve the conflict before launching more compute.

## Directory contract

- `launchers/pretraining/launch_sweep.py`: shared validation, spec generation,
  and explicit Beaker submission.
- `launchers/pretraining/manifests/`: one source-of-truth manifest per
  intervention.
- `launchers/pretraining/generated/`: disposable rendered Beaker specs.
- `launchers/validation/launch_backfills.py`: shared final-checkpoint eval-only
  rendering and explicit Beaker submission.
- `launchers/validation/manifests/`: exact source checkpoints that require
  post-training validation.
- `plot_pretraining_wave.py`: shared pretraining-loss plots and result tables.
- `plots/pretraining/<intervention>/`: all model-size plots for one
  intervention, including per-size U-plots where LR sweeps exist, the strict
  optimal-LR summary, and fixed-LR scale-transfer comparisons.
- `results/pretraining/<intervention>/results.{json,md}`: exact plotted values,
  completion/selection mode, and W&B links across every registered model size.
- `NEXT_EXPERIMENTS.md`: ordered architecture experiment queue and decisions.
- `RUNS.md`: live v2 launch ledger. `v1/` remains historical provenance.

Do not create another independent Beaker-submission implementation for each
intervention. Add a manifest and an optional thin wrapper around
`launch_sweep.py`.

## Scientific comparison contract

1. The default baseline for new work is the 275M-active wide integration model.
   Successor interventions may additionally show their immediate architectural
   parent when explicitly requested. The geometry-matched hybrid therefore
   compares against both wide integration and the first `expand_v=1` hybrid.
2. An isolated intervention is trained from scratch using the wide recipe plus
   only the named architectural change. Do not initialize an architecture
   ablation from a wide checkpoint.
3. Keep active parameter count close enough for a meaningful comparison and
   report the exact difference. The training token budget is derived from
   active non-embedding parameters, so equal Cx labels may have slightly
   different absolute token counts.
4. Test interventions at Cx1/Cx2/Cx4/Cx8. Early evidence can stop promotion to
   larger model sizes, but it does not retroactively change completed results.
5. Combine interventions only after their isolated tests are understood. The
   combined recipe is a new intervention with its own manifest and plots.

## Canonical 275M pretraining systems settings

All sequence counts below use 8,192-token sequences. The optimizer batch is a
comparability invariant; GPU count, rank microbatch, and accumulation may be
tuned only if their product preserves it.

| Cx | Global tokens | Global sequences | Validated GPUs | Rank microbatch | Accumulation |
|---:|---:|---:|---:|---:|---:|
| 1 | 262,144 | 32 | 2 B300 | 16 sequences | 1 |
| 2 | 393,216 | 48 | 2 B300 | 8 sequences | 3 |
| 4 | 524,288 | 64 | 2 B300 | 16 sequences | 2 |
| 8 | 786,432 | 96 | 2 B300 | 16 sequences | 3 |

The launcher must verify:

```text
global_tokens = sequence_length * world_size * rank_microbatch_sequences * accumulation
```

- Use DDP with EP=1, TP=1, PP=1, and CP=1 by default. Minimize expert
  parallelism; add it only when the model cannot fit or measured throughput
  makes it necessary.
- Current architecture sweeps use two Holmes B300s per run, the latest validated
  image in the manifest, urgent priority, and workspace
  `ai2/OLMo-3-moe-experiments`.
- Pretraining uses `OLMo_mix_0925`, the Dolma 2 tokenizer, 10% token warmup, and
  cosine decay to 10% of peak LR unless the intervention explicitly tests one
  of those choices.
- Never silently change batch size, data, schedule, initialization, sequence
  length, or systems parallelism inside an architectural intervention.

## LR-sweep and promotion rules

1. Begin with factor-of-two-spaced LRs around the wide baseline optimum. Four
   points are appropriate for the first 275M test; later tests may start with
   three when transfer is reliable.
2. Use completed full runs for formal selection. Running losses may guide a
   follow-up launch but never enter the U-plot or summary.
3. A Cx is bracketed only when its lowest observed finished loss is strictly
   inside the swept LR range and the local quadratic fit in `log10(LR)` is
   convex with its fitted minimum inside the observed range.
4. If the observed best is an edge point, launch another point in the improving
   direction. Do not publish an optimal LR for that Cx yet.
5. The selected LR and loss are always the best observed finished point. The
   quadratic prediction is a diagnostic and may guide follow-ups; it is never
   substituted into the summary as if it were trained.
6. Use the final-250M-token mean training CE as the primary pretraining metric.
   The formal plotter must verify that the registered W&B history spans the
   complete 250M-token interval; never publish a partial final-window mean. If
   a W&B reset or resume creates a shorter final segment, stop regeneration and
   combine the predecessor run history before plotting. Check wider/narrower
   windows separately when differences are small or noisy.

## Plotting contract

- Produce exactly one all-Cx U-plot per intervention. It has one intervention
  curve per Cx. Ordinarily it is intervention-only; an explicitly requested
  parent comparison may add reference markers without adding reference curves.
- Produce exactly one observed-best summary per intervention. It contains the
  intervention and wide baseline, plus any explicitly requested immediate
  parent, but only for Cx values satisfying the bracketing rule above.
- Larger-size runs launched only at transferred wide-optimal LRs belong in a
  separate fixed-LR comparison plot. Never label them hybrid-optimal until a
  hybrid LR sweep brackets an observed best.
- Annotate the summary with actual observed-best LRs. Fitted stars and vertical
  lines belong only on the U-plot.
- Do not produce delta plots by default.
- Use exact registered W&B run IDs, not broad name matches. Exclude smoke,
  failed, stopped, diagnostic, noncanonical-batch, and running points.
- Reuse `.cache/jacobm_olmoe_ladder/v1/wandb_histories`; do not repeatedly
  download complete W&B histories.
- Store both JSON and Markdown result tables next to the plot family. Tables may
  show running/provisional points, but must label them clearly.
- New pretraining results compare with wide integration, not every v1
  intervention. The geometry-matched hybrid also includes the first hybrid as
  its immediate parent reference by explicit project decision.
- Whenever plots or result tables change, update the run ledger in the same
  commit when applicable and push all three artifact types to the current
  GitHub experiment branch.

## Launch workflow

1. Write the intervention hypothesis and enumerate every change from wide.
2. Implement the model/trainer and verify parameter counts and layer placement.
3. Run a short smoke for a new code path. A new LR point using already validated
   code does not require another smoke.
4. Create or update the intervention manifest.
5. Dry-render the exact points and inspect the printed batch equations, run
   names, GPU total, source repo, image, workspace, cluster, and priority.
6. Inspect the rendered YAML under `launchers/pretraining/generated/`.
7. Submit only with explicit `--submit --experiment-name ...`.
8. Confirm Beaker `Started`, the intended step/token budget, W&B initialization,
   and at least one real optimizer step. Check all tasks for early errors.
9. Record the Beaker experiment, every job ID, every W&B ID, settings, and status
   in `RUNS.md`. Add W&B IDs to the plotting registry.
10. Regenerate plots after completion; extend any unbracketed Cx before making a
    promotion decision.

One Beaker experiment should normally contain all tasks for one coherent
intervention wave. Use exact `--point CX:LR` selectors for targeted extensions.

## Naming, checkpoints, and resumption

- Run names are semantic and resume-stable:

  ```text
  pt-<size>-<intervention>-cx<Cx>-lr<LR-tag>-r<replica>
  ```

- Systems-only details do not belong in the run name; they belong in the
  manifest and run ledger.
- A new scientific run gets a new semantic run name. A retry/resume keeps the
  original name and checkpoint directory.
- Prefer resuming the original Beaker experiment. The launcher refuses selected
  run names whose checkpoint directories already exist unless
  `--resume-existing` is explicitly supplied.
- Store DDP pretraining checkpoints under
  `/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmo-ddp/pretraining/`.
- Match v1 resume retention for full training: write an ephemeral checkpoint
  every 500 steps with `remove=ephemeral_only`, so each new resume checkpoint
  replaces the previous ephemeral one. Retain the final checkpoint permanently;
  Jacob handles cleanup of permanent checkpoints separately.

## Trainer callback contract

Every new v2 pretraining, midtraining, and long-context trainer must include:

- `SpeedMonitorCallback` for rolling MFU;
- `BeakerCallback` so the job description exposes training progress and MFU;
- `WandBCallback` and `ConfigSaverCallback`;
- checkpointing appropriate to the stage.

Do not enable language-model or downstream evaluator callbacks in a training
job, including at the final training step. In-loop evaluation has caused both
large throughput regressions and illegal-memory failures in the DDP training
process. Run validation in a separate eval-only job from the completed final
checkpoint instead.

Pretraining plots use training CE plus post-training validation results.
Midtraining and long-context comparisons use post-training validation rather
than training-loss plots. Long-context additionally uses external RULER
through HF/vLLM on Jupiter.

## Completion and promotion record

For each intervention, record:

- exact architecture delta and active/total parameter counts;
- manifest revision and training entrypoint;
- complete per-Cx LR grid and canonical batch settings;
- Beaker and W&B identities;
- observed-best losses/LRs and whether every Cx is bracketed;
- post-training validation results and eval-job identity;
- decision: reject, retain for combination, or gather more evidence.

Do not declare an intervention promoted from a partial run, predicted LR, or
unbracketed edge point.
