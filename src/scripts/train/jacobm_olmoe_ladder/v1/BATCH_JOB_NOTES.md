# Batch Job Notes

This file records reusable Beaker patterns for ladder maintenance jobs that are
too slow or flaky from an interactive CPU-only session.

## W&B Plot Refresh

Use a direct Beaker spec with the Weka repo mounted, rather than a Gantry clone,
so local scripts and generated files are exactly the ones in this workspace.

Prefer the narrowest plotter that covers the runs that changed. For example, if
only integration-test runs have finished since the last refresh, run the
integration plotter directly instead of `plot_all_experiments.sh`. This avoids
the legacy baseline plotter path, which can still use full-history scans for
uncached finished baseline runs.

Current integration-only tail refreshes:

- `2026-07-07`: `olmoe3-integration-plot-tail-refresh-20260707-saturn`, Beaker `https://beaker.org/ex/01KWZ6T1AG0R0JHC99F3JBT0JG` (Saturn requeue; old Jupiter copy `01KWZ37HQ4NHRTYZAS5PVGFP9C` was stopped)
- `2026-07-07`: `olmoe3-integration-plot-tail-refresh-20260707b`, Beaker `https://beaker.org/ex/01KWZ37HQ4NHRTYZAS5PVGFP9C` (stopped while queued on Jupiter)
- `2026-07-06`: `olmoe3-integration-plot-tail-refresh`, Beaker `https://beaker.org/ex/01KWV2Z8ZNHKJQBJ7K2QYS3CEM`
- Workspace: `ai2/OLMo-3-moe-experiments`
- Cluster: `ai2/jupiter`
- Resources: 1 GPU, urgent priority
- Repo path: `/weka/oe-adapt-default/jacobm/olmoe3/OLMo-core`
- Python: image system Python, with plotting deps already available

Command:

```bash
SKIP_UNCACHED_FINISHED_HISTORY=0 \
  python src/scripts/train/jacobm_olmoe_ladder/experiments/integration/plot_integration.py
```

`plot_integration.py` calls `scan_history_cached(..., tail_window_tokens=250M)`,
so uncached finished runs are fetched through the tail-history path. Existing
valid caches are still reused.

The previous broad test job was:

- Experiment: `olmoe3-plot-refresh-jupiter-weka-system-python`
- Beaker: `https://beaker.org/ex/01KWV1CQM54AAHA7HK5QYVS682`
- Workspace: `ai2/OLMo-3-moe-experiments`
- Cluster: `ai2/jupiter`
- Resources: 1 GPU, urgent priority
- Repo path: `/weka/oe-adapt-default/jacobm/olmoe3/OLMo-core`
- Python: image system Python, with plotting deps already available

Important details:

- Mount `oe-adapt-default` at `/weka/oe-adapt-default`.
- Mount `oe-training-default` if result scripts may touch checkpoint or eval
  metadata.
- Set `WANDB_API_KEY` from `jacobm_WANDB_API_KEY`.
- Set `MPLBACKEND=Agg`.
- Run from the mounted repo:

```bash
SKIP_UNCACHED_FINISHED_HISTORY=0 REFRESH_STALE_CACHE=0 INCLUDE_RUNNING=0 \
  src/scripts/train/jacobm_olmoe_ladder/experiments/plot_all_experiments.sh
```

The broad refresh uses the experiment plotters' tail-history paths for completed
runs where available, but also runs the baseline plotter. Use it only when
several plot families need to change.

## HF Conversion

Use one Beaker experiment per checkpoint, with one urgent Jupiter GPU. The job
mounts the Weka repo and checkpoint tree, sets this repo's `src` on
`PYTHONPATH`, and runs the existing conversion wrapper:

```bash
python src/scripts/train/jacobm_olmoe_ladder/convert_275m_eval_targets.py \
  --manifest src/scripts/train/jacobm_olmoe_ladder/eval_1p2b_integration_cx1_cx2_targets.jsonl \
  --only <train_run_name> \
  --work-dir /tmp/olmoe3-hf-convert-work \
  --log-dir /results/convert-logs
```

The reusable launcher is:

```bash
/tmp/olmoe-plot-venv/bin/python \
  src/scripts/train/jacobm_olmoe_ladder/launch_hf_conversion_batch.py
```

Current 1.2B integration conversion batch:

- `https://beaker.org/ex/01KWV22KG38G1SMDN54MQVX9WM`
- `https://beaker.org/ex/01KWV22MPAR8DPENAQ03Y8N66S`
- `https://beaker.org/ex/01KWV22P0FKBV20ZKXXNEYHTH5`
- `https://beaker.org/ex/01KWV22Q5JF4CGAR5R1E576NE5`

After conversion, launch OLMoBase evals with:

```bash
/tmp/olmoe-plot-venv/bin/python \
  src/scripts/train/jacobm_olmoe_ladder/launch_olmobase_evals.py \
  --manifest src/scripts/train/jacobm_olmoe_ladder/eval_1p2b_integration_cx1_cx2_targets.jsonl \
  --launch
```

The eval launcher uses 8 single-GPU vLLM engines on Jupiter, urgent priority,
and the same `olmo-eval` image/settings used for the Cx8 eval batch.
