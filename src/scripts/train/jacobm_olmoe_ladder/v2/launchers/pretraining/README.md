# v2 pretraining launchers

`launch_sweep.py` handles same-shape LR sweeps. `launch_hybrid_scale_smokes.py`
handles the heterogeneous GPU/EP microbatch probes needed before promoting a
new model size. Both consume checked-in manifests; rendered Beaker YAML goes
under `generated/` and is not source-controlled.

The launcher is a dry run unless `--submit` and `--experiment-name` are both
provided. It validates the per-Cx optimizer batch, sequence length, world size,
rank microbatch, gradient accumulation, EP settings, unique run names, source
wrapper, and selected LR points before writing a spec.

```bash
# Inspect the complete current hybrid grid without launching.
src/scripts/train/jacobm_olmoe_ladder/v2/launchers/pretraining/launch_275m_hybrid_gdn_ev1.sh

# Render an exact extension set without launching.
src/scripts/train/jacobm_olmoe_ladder/v2/launchers/pretraining/launch_275m_hybrid_gdn_ev1.sh \
  --point 1:4e-4 --point 2:4e-4 --point 4:3.2e-3 --point 8:3.2e-3

# Launch only after inspecting the rendered YAML and task summary.
src/scripts/train/jacobm_olmoe_ladder/v2/launchers/pretraining/launch_275m_hybrid_gdn_ev1.sh \
  --point 4:3.2e-3 \
  --submit --experiment-name jacobm-example-explicit-name

# Inspect the current largest-candidate 480M/810M/1.2B microbatch smokes.
src/scripts/train/jacobm_olmoe_ladder/v2/launchers/pretraining/launch_hybrid_scale_smokes.sh

# Render only the two 1.2B synchronized EP8 probes.
src/scripts/train/jacobm_olmoe_ladder/v2/launchers/pretraining/launch_hybrid_scale_smokes.sh \
  --task 1p2b-cx1-mb8-ep8-sync --task 1p2b-cx2-mb12-ep8-sync

# Inspect the promoted 810M/1.2B Cx4/Cx8 production jobs.
src/scripts/train/jacobm_olmoe_ladder/v2/launchers/pretraining/launch_hybrid_scale_full_cx4_cx8.sh
```

The scale launcher copies the Beaker wrapper to node-local `/tmp` before
executing it, so an in-progress job is isolated from later edits to the shared
source checkout. Scale smokes save one final hard-stop checkpoint; intermediate
checkpoint intervals are intentionally beyond the 12-step smoke horizon.

To add an intervention:

1. Implement and smoke-test its DDP training entrypoint and Beaker wrapper.
2. Copy a manifest under `manifests/`; change the run prefix, wrapper, image if
   needed, environment-variable mapping, active LR grids, and checkpoint root.
3. Run a dry render and verify every printed batch equation.
4. Add the exact W&B IDs to `v2/plot_pretraining_wave.py` after W&B initializes.
5. Record the experiment, job, and W&B IDs in `v2/RUNS.md`.

The existing-checkpoint guard prevents accidental duplicate launches. Do not
override it merely to get a submission through: resume the original experiment
when possible. `--resume-existing` is reserved for an intentional requeue that
must keep the same resume-stable run name and checkpoint directory.
