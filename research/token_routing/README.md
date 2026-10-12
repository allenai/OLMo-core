# research/token_routing — token / KV compaction research (learned routers, soft-token slots)

How much of a long context does a model need to see? This directory holds the experiment code for
token-selection schemes, learned token routers, and soft-token (pooled-doc KV) slot constructions on
the CTC tasks. Code only: result JSONs, logs, run state and router weights stay out of git.

| subdir | what | records |
|---|---|---|
| `devloss_grid/` | answer-CE of a frozen checkpoint under many token-selection schemes (22 CTC rows + cpt80); staging + render. Sibling of `learned_router/` (they import each other) | `devloss-grid-underperformers` |
| `learned_router/` | learned linear token router (REINFORCE / e2e / differentiable), `layerskip/` layer-skip router, `oracle/` labels, `viz/` keep-mask page, `dbg/` | `learned-token-router` |
| `ds64/` | short-heavy 2k-64k data-scaling campaign, dense vs soft-token arms (Beaker orchestration, diagnosis, FLOP audit, scheme grid) | `ds64-scaling-plan`, `ds64-handoff`, `ds64-overnight-2026-09-14` |
| `ds64_fast2k/` | <1 h 2k-row screening loop for soft-token recipes | `ds64-handoff` |
| `ds64_fast8k/` | the same loop at the 8k rung where outlier fails | `ds64-overnight-2026-09-14` |
| `pooled_kv/` | eval-side slot probes (`outlier_probe/`, `landmark_probe/`, `layer_probe/`), trained-parity checks | `soft-kv-slot-probe-handoff`, `pooled-doc-kv-handoff`, `pooled-doc-kv-attention`, `outlier-*-probe` (`outlier-slot-probe`, `outlier-richer-slot-probe`, `outlier-realtoken-parity-probe`, `outlier-saliency-preview-probe`, `outlier-cc00-evaltime-probe`), `trained-parity-winning-arms`, `layer-soft-probe` |
| `pooledkv_train/` | pooled-doc KV trainer, oracle slot cache, eval and bake scripts | `pooled-doc-kv-handoff`, `pooled-doc-kv-attention` |
| `softdetach/` | soft-detached pooling for continued pretraining (CPT) | `softdetach-cpt-plan`, `softdetach-cpt-crossover` |

## Conventions and where things live

* Run scripts from the repo root. Shell/sbatch scripts derive the repo root with `git rev-parse` (or
  `$SLURM_SUBMIT_DIR` under sbatch); Python derives it from `__file__`.
* **Router weights** are not in git: set `ROUTER_WEIGHTS_DIR` / `LAYERSKIP_WEIGHTS_DIR`. Defaults point at
  the original checkout, `/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/learned_router/{weights,layerskip/weights}`.
* **Grid tables** `devloss_grid/grid.json` / `grid.csv` are not ported; originals are at
  `/accounts/projects/berkeleynlp/prasann/projects/OLMo-core/debug/devloss_grid/` (regenerate with `collect_grid.py`).
* Cross-component dependencies (new paths): `recipes/train/research_trainer/beaker_ctc_suite.py`,
  `recipes/eval/legacy_native/{run_q4b_beaker_multirung_eval.py,run_beaker_multirung_eval.sh,ctc_eval}`,
  `recipes/data/convert_unified_to_document_landmark.py`, `recipes/env/local_env.sh`,
  `research/ffn_compute/flop_scaling/beaker_bench_launch.py`, `research/common/corpus_reasoning`
  (put `research/common` on `PYTHONPATH`), the repo's vendored `ctc/`, and `~/projects/olmo-eval` for the vendored CTC specs.
* Known gap: `pooled_kv/landmark_probe/landmark_slot_probe.py` imports `convert_longctx_tasks_to_sft`
  from `src/scripts/data`, which is outside this port map.
