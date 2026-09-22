# cpt/softdetach — soft-detached pooling for continued pretraining (2026-09-21)

Question (Prasann): does CPT with soft-detached pooling at ≥4× compaction reach the same dev loss
as dense CPT at the same FLOP budget? Plan + status: `records/softdetach-cpt-plan.md`.

| file | role |
|---|---|
| `build_cpt_shard.py` / `build_cpt_shard_beaker.sh` | marker-wrapped CPT shards (127 × 512-token pseudo-docs per 64k row) from amandab's tokenized dolma3_longmino sample; train shard (parts 0–7) + a held-out dev shard (last part) — both arms train on the SAME shard |
| `launch_softdetach_cpt.py` | Beaker launcher via `beaker_ctc_suite.py` (urgent, unallocated): `dense`, `sd20` (random 20% of blocks whole, rest one slot), `sfl20` (first_last 20% of every block); same rows/step, `--max-tokens` budgets |
| `eval_cpt_devloss.py` / `eval_cpt_devloss_beaker.sh` | dev loss on the held-out shard: full-row CE and last-20% ("cpt80") CE under full attention, plus the soft arm's own construction |
| `LAUNCH_LEDGER.tsv` | every launch (written by the launcher) |

Screening BEFORE the 4B runs happens on the frozen-base dev-loss harness
(`debug/devloss_grid/ctc_devloss_grid.py --task cpt80`, schemes `rand20 / fl20 / first64 / first128 /
rule20 / grad20 / attnrow20`), see the record. Fractional `--st-keep-token-k` (0 < k < 1 = per-document
fraction) and the `custom` keep rule (`mark_positions_free`) were added for this.
