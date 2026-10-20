# setA Pareto: dense vs bar (gold_fl20p8_noslot) on the full setA mix

Question: is the bar heuristic (soft-token arm `gfl20p8ns`: gold docs whole, 8-token prefix +
first/last 20% of every other body, no slot) Pareto-better than dense SFT on setA, at every
FLOP and wall-clock budget, on all 11 tasks? Experiment state: `ops exp show setA_pareto`;
plan background: `records/flopdrop-final-figures-plan.md` (Set 2).

| File | What |
|---|---|
| `data_beaker.sh` / `_data_inner.sh` | CPU gantry jobs. `gold`: re-tokenize setA (seed 42) with gold sidecars, byte-check against the shipped shards, merge rows ≤ 32,768 tokens. `expand <seed>`: build fresh setA rows (2k–32k buckets, eval-calibrated) at another seed with sidecars, measure overlap with seed 42, merge. |
| `setA_gold_sidecars.py` | Re-runs a built set's converter call (args from its `manifest.json`) with `--emit-gold-sidecar --gold-sidecar-empty-for-missing`; fails if any token/mask byte differs. |
| `expansion_overlap.py` | Per (task, bucket): fraction of a new seed's rows whose question / gold-doc set / whole row already occurs in seed 42 (supply-bound pools re-use gold material). |
| `bench_fla_varlen.py` | 1-GPU timing of a new sequence width in a Qwen3.5 training step: 22–51 s stock (FLA re-autotune), 1–4 s after `freeze_fla_length_autotune`. The cause of the 2026-10-20 fast8k bar-arm stalls. |

Data lands in `/weka/oe-training-default/ai2-llm/checkpoints/prasanns/ctc_sft_sets/setA_pareto/`
(`setA32k_s<seed>/`, `reports/`). Merging uses `ds64_fast8k/merge_tasks.py --max-row-len 32768`.

Why the expansion: the bar arm costs well under half of dense's FLOPs per example, so matching
dense's one-epoch FLOPs needs roughly 2.5–3x setA's rows. Fresh seeds re-draw from the same seed
pools; check `reports/overlap_s<seed>.json` before treating a seed's rows as new data.
