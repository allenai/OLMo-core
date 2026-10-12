# research/ffn_compute — FFN-axis compute reduction and the FLOP-scaling study

Question: how little FFN compute can a token get away with, and does the saving show up as a better
accuracy-per-FLOP frontier than dense SFT at matched training FLOPs? Four pieces, from micro-benchmark
to full campaign.

| dir | what it is | entry points |
|---|---|---|
| `ffnmoe_bench/` | layer- and step-level speed benches/profilers for the nested-width routed FFN (does the FLOP saving become wall-clock?). Code only; the measured JSON/logs are not in git. | `bench_ffn_routing.py`, `profile_routed_layer.py`, `profile_train_step.py`, `smoke_router_learns.py` (CPU) |
| `ffnmoe_train/` | nested-width FFN MoE training family on Qwen3-4B (learned router + budget hinge loss): base baking, local/slurm training, contradiction eval. Its README is the detailed write-up. | `bake_ffn_moe_into_base.py`, `Qwen3-4B-ffnmoe-contra-local.py`, `run_q4b_ffnmoe.sbatch`, `eval_q4b_ffnmoe_contra.sbatch`, `setup_ffnmoe_assets.sbatch` |
| `ffndrop/` | drop-CPT: continued pretraining with random per-token FFN dropping (null expert, no router), then routed-FFN SFT. | `launch_ffndrop_cpt.py`, `pipeline.py`, `eval_drop_devloss*.{py,sh}`, `chain_evals.sh` |
| `flop_scaling/` | the FLOP-scaling campaign (dense vs ffnmoe vs KV-keep arms, short-heavy length mix, 4 tasks, Qwen3.5-4B track + model-scale ladder + Qwen3 arms): launchers, orchestrators, collectors, fits, report/plot builders. | `launch_grid35.py`, `orchestrate35.py`, `collect_results35.py`, `fit_scaling.py`, `make_report.py`, `make_axes_plots.py`, `collect_walltime.py` |

The KV-route / block-skip smoke and bench scripts that used to sit in `debug/flop_scaling/` live in
`research/kv_route/smoke/`, not here.

## Records

- [`records/flop-scaling-ffn-kv-plan.md`](../../records/flop-scaling-ffn-kv-plan.md) — design and status of the campaign (flop_scaling).
- [`records/flop-scaling-report-2026-09-02.md`](../../records/flop-scaling-report-2026-09-02.md) — the generated report (`make_report.py`).
- [`records/ffndrop-cpt-plan.md`](../../records/ffndrop-cpt-plan.md) — drop-CPT plan and results (ffndrop).

## Cross-directory dependencies (live elsewhere in the repo)

- Trainer/launchers: `recipes/train/research_trainer/` (`train_ctc_suite.py`, `beaker_ctc_suite.py`).
- Native evals: `recipes/eval/legacy_native/` (`run_q4b_beaker_multirung_eval.py`, package `ctc_eval`); the
  ffnmoe sbatch evals call `research/common/corpus_reasoning/eval/`. Put `research/common` and
  `recipes/eval/legacy_native` on `PYTHONPATH` (the launchers here already do).
- `research/common/beaker_native_lengthmix_eval.py` (lengthmix eval wrapper) and
  `research/common/taskscale_points.json` (prior dense points read by `collect_results35.py`).
- `research/token_routing/softdetach/eval_cpt_devloss.py` (imported by `ffndrop/eval_drop_devloss.py`).
- Data conversion: `recipes/data/convert_unified_to_document_landmark.py`, `recipes/data/fix_marker_embeddings_qwen35.py`;
  `ctc/` for the short-heavy data builds (`build_shortheavy_data.sbatch`, `CTC_SRC` defaults to `$REPO/ctc/src`).

## Paths and knobs

- Repo root is derived from `__file__` (Python) or `REPO_ROOT` / `git rev-parse --show-toplevel` (shell; under
  `sbatch` it falls back to `SLURM_SUBMIT_DIR`, so submit from inside the checkout or export `REPO_ROOT`).
- `#SBATCH --output=/data/prasann/...` lines cannot expand env vars; override at submit time with
  `sbatch --output=...`. In-script log dirs use `LOG_DIR` / `FFNDROP_LOG_DIR` (default `/data/prasann/logs`).
- `ffndrop`: `FFNDROP_CONDA_BIN`, `FFNDROP_STATE`. `flop_scaling`: `FLOP_SCALING_VIZ` (plot output dir, default
  `<repo>/visualizations/flop_scaling`), `FS35_BASE`, `FS35_BASE_TAG`.

## What is kept and what lives outside git

Kept: code, READMEs, launch ledgers (`ffnmoe_train/`, `ffndrop/`, `flop_scaling/LAUNCH_LEDGER.tsv`),
`flop_scaling/dense_*.tsv` and `prior_dense_state.json` (read by `collect_walltime.py` / `make_axes_plots.py`),
`summary35.md`. Dropped: `orchestrate*_state.json` (re-created by the orchestrators), orchestrator logs,
`launch_logs/`, `_probe/`, bench JSON/kernel dumps. Results: Beaker result datasets and
`results/flop_scaling/*.csv` (regenerate with the collectors); checkpoints on weka
(`.../prasanns/ctc_suite/ckpts`, `flop_scaling35/`, `ffndrop_cpt/`) and node-local `/data/prasann/{ffnmoe_exp,flop_scaling}`.
