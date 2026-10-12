# ds64 — short-heavy 2k-64k data-scaling campaign (dense vs soft-token / pooled-doc KV)

Beaker-driven campaign comparing dense training against soft-token (pooled-doc KV) constructions on
outlier / contradiction / nq / oolong with Qwen3.5-4B (27B for a subset). Plan and handoff:
`records/ds64-scaling-plan.md`, `records/ds64-handoff.md`, `records/ds64-overnight-2026-09-14.md`.

| file | role |
|---|---|
| `build_ds64_data_beaker.sh` | gantry data build: per-rung pools via `ctc-data` (the repo's vendored `ctc/`, installed with `pip install -e $REPO_ROOT/ctc`), nested short-heavy arms, marker tokenization via `recipes/data/convert_unified_to_document_landmark.py`, written to weka |
| `compose_uniform_arms.py` | nested-prefix arm composer used by the build |
| `launch_ds64.py` | arm definitions (`ARM_EXTRA`) and Beaker launcher via `recipes/train/research_trainer/beaker_ctc_suite.py`; writes `LAUNCH_LEDGER.tsv` |
| `orchestrate_ds64.py` | unattended train -> eval -> harvest orchestrator (state `orchestrator_ds64_state.json`, not in git); reads `data_build_jobs.tsv`, `soft_arms.json` |
| `collect_ds64.py`, `harvest_ds64.sh`, `make_pareto_artifact.py` | gather ladder evals + FLOP meter into `results/ds64/results.csv`, render the Pareto page |
| `launch_ns_evals.py` + `ns_evals.tsv` | evals for the slot-less (`*ns`) arms, same command as the orchestrator |
| `reset_failed_arms.py` | drop FAILED runs from the orchestrator state so they relaunch |
| `probe_outlier_header.py`, `probe_outlier_keepset.py`, `repro_firstk_compaction.py`, `repro_xhdr_compaction.py`, `xhdr_diag_weka.sh`, `xhdr_collapse_diagnosis.md` | outlier collapse diagnosis (CPU repro of what the trainer builds) |
| `flop_audit/` | FLOP-meter discrepancy audit (see its README) |
| `scheme_grid/` | method x task grid (`build_grid.py`, `grid.json`, `render.py`) |

Not in git: `launch_logs/`, `xhdr_logs/`, orchestrator state/logs and `.bak-*` snapshots, `results/ds64/`
(all remain in the original checkout under `debug/ds64/`). Fast screening loops built on the same
machinery: `../ds64_fast2k`, `../ds64_fast8k`. Eval-side probes of the trained arms: `../pooled_kv`.
