# flop_scaling

The FLOP-scaling campaign (plan: `records/flop-scaling-ffn-kv-plan.md`, report: `records/flop-scaling-report-2026-09-02.md`).
Beaker-driven; orchestrators run detached on the login node and keep resumable state in `orchestrate*_state.json`
(not in git).

- Qwen3.5-4B track: `launch_grid35.py` (arms/budgets), `orchestrate35.py` (train -> eval chain), `collect_results35.py`,
  `build_kv35_shards*`, `build_shortheavy_data*`, `compose_shortheavy_arms.py`, `prep_bases_beaker.sh`.
- Older Qwen3 grid and model-scale ladder: `launch_grid.py`, `orchestrate.py`, `orchestrate_scale.py`, `collect_results.py`,
  `collect_scale.py`, `scale_multipliers.py`.
- Analysis/report: `fit_scaling.py`, `make_report.py`, `make_axes_plots.py`, `collect_walltime.py`, `qwen35_memory.py`,
  `qwen35_shares.py`; synced via `harvest_to_s3.sh`, `sync_*.sh`, `upload_base_to_s3.sbatch`.
- Bench: `bench_ffn_speed.py`, `beaker_bench_launch.py`, `profile_softtoken35.py`, `beaker_status.py`, `launch_evals.py`.
- Provenance kept: `LAUNCH_LEDGER.tsv`, `dense_anchor_launches.tsv`, `dense_extra_evals.tsv`, `prior_dense_state.json`
  (`collect_walltime.py` reads it), `summary35.md`.

Inputs elsewhere: `research/common/taskscale_points.json`, `research/common/beaker_native_lengthmix_eval.py`,
`recipes/train/research_trainer/beaker_ctc_suite.py`, `recipes/eval/legacy_native/run_q4b_beaker_multirung_eval.py`.
KV-route/block-skip smokes that were here are in `research/kv_route/smoke/`.
Results (`results/flop_scaling/*.csv`, plots under `visualizations/flop_scaling` or `$FLOP_SCALING_VIZ`) are regenerated
by the collectors and not stored in git.
