# sft_xlong256k/ — 256k-context SFT on the xlong5 2k→256k ladder (Qwen3.5)

SFT of the Qwen3.5-4B 256k CPT models on 75% the xlong5 2k→256k 5-task mix / 25%
`allenai/Dolci-Instruct-SFT`, at a 262,144 window. Beaker only.

| Script | Arm | Data root | Batch / LR | Control for |
|---|---|---|---|---|
| `Qwen3.5-4B-dense-xlong5-qafter-dolci25-256k-SFT.py` | dense (GDN hybrid, full-attn blocks untouched) | `xlong5_2k256k_qwen35_qafter/shards_full` | 2 nodes, CP=4 → DP=4 × 262144 = 1.05M tok/step, LR 4e-5 | Pilot on the **query-after** data build. Token-matched (2.35B) to `q35-4b-dense-xlong5-dolci25-256k`, but see the caveat below — it is **not** a clean control for it. |

## Relationship to the two legacy 256k scripts

The dense/landmark 256k pair this family grew out of still lives at
`src/scripts/train/sft/amanda-landmark/`:

- `Qwen3.5-4B-dense-xlong5-dolci25-256k-SFT.py` (run `q35-4b-dense-xlong5-dolci25-256k`)
- `Qwen3.5-4B-fast-compressive-landmark-xlong5-dolci25-256k-SFT.py`

They are token-matched to each other at 560 steps × 4.19M tokens and remain a valid
landmark-vs-dense pair. New 256k scripts go here instead, per CLAUDE.md.

## Caveat on the qafter comparison

The qafter arm changes **three** things against `q35-4b-dense-xlong5-dolci25-256k`, not one:

1. the data build (`both` → `after` query position),
2. the global batch (4.19M → 1.05M tokens/step, via 8 nodes → 2 nodes at a fixed CP=4),
3. the LR (1e-5 → 4e-5, sqrt-scaled from the 32k SFT family's 65,536-token anchor, on the
   view that the legacy run's unscaled 1e-5 at a 64× batch was mis-set).

(2) and (3) together are ~16× the legacy run's path length through parameter space, so a
qafter-vs-standard delta off these two runs is confounded. Treat the qafter run as a pilot; a
clean pair needs a standard-data arm rerun at this batch and LR.

Also, from the qafter tree's own README: **outlier is a deliberate no-op** between the two data
builds (its converter branch was already query-after), so it must be excluded from any
query-position ablation — four tasks change, not five. And eval must be launched with
`--query-position after` against `xlong5_2k256k_qwen35/eval/`, since the qafter root ships no
`eval/` directory and query position is an eval-time rendering flag.
