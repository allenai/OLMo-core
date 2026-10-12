# ffndrop — drop-CPT: random null-FFN CPT, then routed-FFN SFT (2026-09-29)

Question (Prasann): the learned FFN router was never compute-optimal against dense SFT. Does a CPT stage
that randomly drops FFNs per token (AdaMoE-style null expert, no router, no budget loss) make the routed
SFT stage compute-optimal at matched SFT FLOPs? Plan + status: `records/ffndrop-cpt-plan.md`.

| file | role |
|---|---|
| `launch_ffndrop_cpt.py` | Beaker launcher via `recipes/train/research_trainer/beaker_ctc_suite.py` (urgent, unallocated): `dense` control vs `drop75l12`, on the soft-detach CPT shards (`softdetach_cpt/shards/cpt_u1B`), same base/rows/LR as `research/token_routing/softdetach/` |
| `eval_drop_devloss.py` / `eval_drop_devloss_beaker.sh` | drop-robustness dev loss: body-token CE on `cpt_dev` with FFNs (layers >= 1) skipped per token at fixed r = 0/0.25/0.5/0.75; same pattern for every checkpoint (paired) -> weka `ffndrop_cpt/devloss/<name>.json` |
| `pipeline.py` | login-node chain: wait for the CPT run -> dev-loss eval -> routed-FFN SFT via `../flop_scaling/launch_grid35.py` -> native ladder eval via `research/common/beaker_native_lengthmix_eval.py`. Resumable state file (`pipeline_state.json`, untracked; override with `FFNDROP_STATE`). |
| `chain_evals.sh` | login-node watcher that submits the dev-loss eval once a CPT run exits |
| `LAUNCH_LEDGER.tsv` | every launch (written by the launcher) |

Env knobs: `FFNDROP_CONDA_BIN` (conda `bin/` put on PATH by `pipeline.py` / `chain_evals.sh`; default
`/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin`), `FFNDROP_LOG_DIR` (default
`/data/prasann/logs`), `REPO_ROOT`. `eval_drop_devloss.py` imports `eval_cpt_devloss.py` from the sibling
`research/token_routing/softdetach/` (not copied here).

Mechanism: `olmo_core.nn.ffn_token_drop` (`Transformer.enable_ffn_token_drop`), trainer flags
`--ffn-drop-max-rate / --ffn-drop-layer-prob / --ffn-drop-start-layer` on `train_ctc_suite.py`
(training-only; the export is a plain dense checkpoint). Realized drop fraction: wandb `ffn_drop/frac`.
Dev loss: `research/token_routing/softdetach/eval_cpt_devloss_beaker.sh` on the held-out `cpt_dev` shard.
