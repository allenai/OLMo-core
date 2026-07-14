# Jacob's OLMoE ladder

This directory is the DDP-branch home for Jacob's MoE architecture ladder.

- [`v1/`](v1/) is the complete legacy experiment package copied at the 2026-07-14 cutover. It contains the original launchers, plans, ledgers, generated results, plots, eval target lists, and analysis tools.
- [`v1/MIGRATION.md`](v1/MIGRATION.md) records the exact source revision, snapshot scope, intentional experiment omissions, and cutover boundary.
- [`v1/DEFAULT_RUN_SETTINGS.md`](v1/DEFAULT_RUN_SETTINGS.md) is the compact settings handoff; `v1/SETTINGS_AUDIT.md` remains the detailed historical source.
- [`v1/ARTIFACTS.md`](v1/ARTIFACTS.md) defines the local and GCS artifact layout.

The launchers under `v1/` target the legacy non-DDP training API and are retained for provenance and reproduction only. Clean DDP pretraining, midtraining, and long-context launchers will be added here after their configs are smoke-tested. Until then, do not launch a new training run from a legacy script by accident.

The current W&B project remains `ai2-llm/jacobm-olmoe-ladder`. The migrated history cache defaults to `.cache/jacobm_olmoe_ladder/v1/wandb_histories` at the repository root and can be overridden with `OLMOE3_WANDB_CACHE_DIR`.
