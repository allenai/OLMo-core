# cpt/ffndrop — drop-CPT: random null-FFN CPT, then routed-FFN SFT (2026-09-29)

Question (Prasann): the learned FFN router was never compute-optimal against dense SFT. Does a CPT stage
that randomly drops FFNs per token (AdaMoE-style null expert, no router, no budget loss) make the routed
SFT stage compute-optimal at matched SFT FLOPs? Plan + status: `records/ffndrop-cpt-plan.md`.

| file | role |
|---|---|
| `launch_ffndrop_cpt.py` | Beaker launcher via `beaker_ctc_suite.py` (urgent, unallocated): `dense` control vs `drop75l12`, on the soft-detach CPT shards (`softdetach_cpt/shards/cpt_u1B`), same base/rows/LR as `cpt/softdetach/` |
| `LAUNCH_LEDGER.tsv` | every launch (written by the launcher) |

Mechanism: `olmo_core.nn.ffn_token_drop` (`Transformer.enable_ffn_token_drop`), trainer flags
`--ffn-drop-max-rate / --ffn-drop-layer-prob / --ffn-drop-start-layer` on `train_ctc_suite.py`
(training-only; the export is a plain dense checkpoint). Realized drop fraction: wandb `ffn_drop/frac`.
Dev loss: `cpt/softdetach/eval_cpt_devloss_beaker.sh` on the held-out `cpt_dev` shard.
