# v2 pretraining launchers

`launch_sweep.py` handles same-shape LR sweeps. `launch_hybrid_scale_smokes.py`
handles the heterogeneous GPU/EP microbatch probes needed before promoting a
new model size. Both consume checked-in manifests; rendered Beaker YAML goes
under `generated/` and is not source-controlled.

The launcher is a dry run unless `--submit` and `--experiment-name` are both
provided. It validates the per-Cx optimizer batch, sequence length, world size,
rank microbatch, gradient accumulation, EP settings, unique run names, source
wrapper, and selected LR points before writing a spec.

For these DDP+EP jobs, the data-parallel world size remains the full GPU world
size; EP creates a separate MoE mesh and does not reduce the number of data
shards. A configured rank microbatch larger than the rank batch is only a cap.
The launcher prints both the cap and effective microbatch and computes
accumulation from the effective value.

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

# Inspect the largest-legal-microbatch geometry/expand_v=2 smokes.
src/scripts/train/jacobm_olmoe_ladder/v2/launchers/pretraining/launch_275m_geometry_gdn_ev2_smokes.sh \
  --task cx1-mb16 --task cx2-mb24 --task cx4-mb32 --task cx8-mb48

# Inspect the 16-point geometry/expand_v=2 LR sweep without launching it.
src/scripts/train/jacobm_olmoe_ladder/v2/launchers/pretraining/launch_275m_geometry_gdn_ev2.sh

# Inspect the checkpoint-free NoPE + attention-gating capacity smokes.
src/scripts/train/jacobm_olmoe_ladder/v2/launchers/pretraining/launch_275m_geometry_gdn_ev2_nope_gated_smokes.sh

# Inspect the 16-point NoPE + attention-gating LR sweep without launching it.
src/scripts/train/jacobm_olmoe_ladder/v2/launchers/pretraining/launch_275m_geometry_gdn_ev2_nope_gated.sh

# Inspect the complete 480M/810M/1.2B NoPE production wave.
PYTHONPATH=src .venv/bin/python \
  src/scripts/train/jacobm_olmoe_ladder/v2/launchers/pretraining/launch_geometry_matched_scale_full.py \
  --manifest src/scripts/train/jacobm_olmoe_ladder/v2/launchers/pretraining/manifests/geometry_matched_scale_nope_full.yaml

# Inspect the equivalent gated-attention wave. This command is deliberately
# dry-run only until the 275M gating sweep establishes the intervention.
PYTHONPATH=src .venv/bin/python \
  src/scripts/train/jacobm_olmoe_ladder/v2/launchers/pretraining/launch_geometry_matched_scale_full.py \
  --manifest src/scripts/train/jacobm_olmoe_ladder/v2/launchers/pretraining/manifests/geometry_matched_scale_nope_gated_full.yaml

# Inspect the compact, ordered RoPE + gated-attention scale wave. It reuses the
# first-hybrid GPU layouts for 480M/810M and the selected 8/16/16/32-GPU 1.2B
# layout. Pass --submit only after the printed 124-GPU batch audit is correct.
src/scripts/train/jacobm_olmoe_ladder/v2/launchers/pretraining/launch_geometry_matched_scale_rope_gated_full.sh

# Render only the two 1.2B synchronized EP8 probes.
src/scripts/train/jacobm_olmoe_ladder/v2/launchers/pretraining/launch_hybrid_scale_smokes.sh \
  --task 1p2b-cx1-mb8-ep8-sync --task 1p2b-cx2-mb12-ep8-sync

# Inspect the promoted 810M/1.2B Cx4/Cx8 production jobs.
src/scripts/train/jacobm_olmoe_ladder/v2/launchers/pretraining/launch_hybrid_scale_full_cx4_cx8.sh

# Inspect the 480M Cx4/Cx8 completion jobs.
src/scripts/train/jacobm_olmoe_ladder/v2/launchers/pretraining/launch_hybrid_scale_480m_cx4_cx8.sh

# Render the canonical GDN2 and KDA sweeps in Cx1/Cx2/Cx4/Cx8 order. The
# launcher reuses, and therefore omits, the existing GDN2 Cx8/1.6e-3 cell.
src/scripts/train/jacobm_olmoe_ladder/v2/launchers/pretraining/launch_275m_canonical_gdn2_kda_lr_sweeps.sh

# Render the four-cell 480M canonical-KDA stability transfer; add --submit to launch.
src/scripts/train/jacobm_olmoe_ladder/v2/launchers/pretraining/launch_480m_geometry_kda_full.sh

# Render the distinct KDA expand_v=2 + negative-eigenvalue fixed-LR transfers.
src/scripts/train/jacobm_olmoe_ladder/v2/launchers/pretraining/launch_275m_kda_ev2_neg_transfer.sh
src/scripts/train/jacobm_olmoe_ladder/v2/launchers/pretraining/launch_480m_kda_ev2_neg_transfer.sh
src/scripts/train/jacobm_olmoe_ladder/v2/launchers/pretraining/launch_810m_kda_ev2_neg_transfer.sh

# Inspect the canonical expand_v=1, nonnegative GDN2 scale transfer. The
# checked-in manifest preserves the approved longest-first submission order
# and uses the balanced 176-GPU layout at full concurrency.
src/scripts/train/jacobm_olmoe_ladder/v2/launchers/pretraining/launch_geometry_matched_scale_gdn2_canonical_full.sh
```

The scale launcher copies the Beaker wrapper to node-local `/tmp` before
executing it, so an in-progress job is isolated from later edits to the shared
source checkout. Scale smokes save one final hard-stop checkpoint; intermediate
checkpoint intervals are intentionally beyond the 12-step smoke horizon.
The geometry/`expand_v=2` and NoPE-plus-gating capacity smokes are deliberate
exceptions: they set the trainer's `no_checkpoints` mode and write no model or
optimizer checkpoint.

Only explicitly designated geometry/`expand_v=2` work uses Beaker's
unallocated queue (`minRuntime: 0m`, `autoResume: true`). This currently
includes the 275M geometry, NoPE, gated-attention, canonical GDN2, and KDA
sweeps plus the larger NoPE production wave. Do not copy that scheduling
exception into other
intervention manifests without an explicit decision. Gantry receives
`preemptible=False`; it omits the deprecated Beaker field while retaining the
requested non-preemptible behavior.

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
