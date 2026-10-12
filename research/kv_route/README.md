# research/kv_route — KV-route / block-skip routers and the flexible-compute speedup question

Three routers (FFN width, KV keep, block skip) plus a joint budget, and the question of how much
training/inference wall-clock they can actually buy ("can the routers reach a 40x speedup?"). The
conceptual write-up is [`records/kv-route-attention-allocation.md`](../../records/kv-route-attention-allocation.md).

| dir | what it is | docs |
|---|---|---|
| `flexcompute_40x/` | FLOP-ceiling and hardware benches: LM-head gating, GDN kernel levers, FFN linearity/share, step partition, router-matrix and joint sweeps, block-skip checks. Code + `rungs_v12-*.json` (router rung assignments used by `mlskip_static_prune.py`). | [`flexcompute_40x/README.md`](flexcompute_40x/README.md) |
| `flexcompute_train/` | training the routed arms: tokenize contradiction for Qwen3.5, train ladder, eval arms, collect. | [`flexcompute_train/README.md`](flexcompute_train/README.md) |
| `viz/` | builds the two published flexcompute pages (`build_flexcompute_pages.py`) and the pages themselves. | `viz/README.md` |
| `smoke/` | GPU smoke and bench scripts for the KV-route and block-skip layers (from the old `debug/flop_scaling/`): `smoke_kv_route_gpu`, `smoke_block_skip_gpu`, `smoke_blockmask_gpu`, `bench_kv_route_layer`, `bench_kv_route_mask`, `mem_profile_routers`, `beaker_pytest_launch`. | `smoke/README.md` |

Records: `records/kv-route-attention-allocation.md` plus the two flexcompute READMEs above. The routed arms'
accuracy-vs-FLOP comparison against dense is the campaign in `research/ffn_compute/flop_scaling/`.

## Paths and knobs

- Repo root: `REPO_ROOT` or `git rev-parse --show-toplevel` (falls back to `SLURM_SUBMIT_DIR` under `sbatch`; submit from
  inside the checkout or export `REPO_ROOT`). Python scripts derive the root from `__file__`.
- Nearly every sbatch writes `#SBATCH --output=/data/prasann/logs/...`; directives cannot expand env vars, so override
  with `sbatch --output=...`. In-script log dirs use `LOG_DIR` (default `/data/prasann/logs`).
- Benches `source /data/prasann/node_local_env.sh` (node-local env) and expect to run on a Berkeley H200 node.
- Training/eval deps: `recipes/train/research_trainer/` (trainer), `recipes/eval/legacy_native/` (`ctc_eval`),
  `recipes/data/convert_unified_to_document_landmark.py`; `PYTHONPATH` is set by the sbatch files.

## Results are not in git

Every `*.json` in `flexcompute_40x/` other than `rungs_v12-*.json` is a measured result (ceiling, bench_*, joint_sweep_*,
router_matrix_*, graph_probe_*, ffn_*...) and was dropped, as were `joint_sweep_*_shell.log` and `__pycache__`. Scripts
write them back into `flexcompute_40x/` when re-run; `viz/build_flexcompute_pages.py` and
`flexcompute_40x/make_*.py` need those JSONs to regenerate the pages (the checked-in HTML pages are the rendered snapshot).
Training checkpoints/evals follow `flexcompute_train/LAUNCH_LEDGER.tsv` (weka / `/data/prasann/...`).
Provenance note: these three directories were untracked on the source branch; they were ported from the working copy.
