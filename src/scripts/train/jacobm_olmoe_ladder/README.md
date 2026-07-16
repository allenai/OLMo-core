# Jacob's OLMoE ladder

This directory is the DDP-branch home for Jacob's MoE architecture ladder.

- [`v1/`](v1/) is the complete legacy experiment package copied at the 2026-07-14 cutover. It contains the original launchers, plans, ledgers, generated results, plots, eval target lists, and analysis tools.
- [`v1/MIGRATION.md`](v1/MIGRATION.md) records the exact source revision, snapshot scope, intentional experiment omissions, and cutover boundary.
- [`v1/DEFAULT_RUN_SETTINGS.md`](v1/DEFAULT_RUN_SETTINGS.md) is the compact settings handoff; `v1/SETTINGS_AUDIT.md` remains the detailed historical source.
- [`v1/ARTIFACTS.md`](v1/ARTIFACTS.md) defines the local and GCS artifact layout.
- [`v2/EXPERIMENT_RULES.md`](v2/EXPERIMENT_RULES.md) is the master contract for
  new DDP experiments and plots.
- [`v2/launchers/pretraining/`](v2/launchers/pretraining/) contains the reusable
  manifest-driven pretraining launcher.
- [`v2/RUNS.md`](v2/RUNS.md) is the live post-migration run ledger.

The legacy launchers under `v1/` are retained for provenance and reproduction
only. New pretraining sweeps use the v2 manifest-driven launcher. Clean shared
midtraining and long-context launchers will be added after their configs are
smoke-tested. Do not launch a new training run from a legacy script by accident.

The current W&B project remains `ai2-llm/jacobm-olmoe-ladder`. The migrated history cache defaults to `.cache/jacobm_olmoe_ladder/v1/wandb_histories` at the repository root and can be overridden with `OLMOE3_WANDB_CACHE_DIR`.
