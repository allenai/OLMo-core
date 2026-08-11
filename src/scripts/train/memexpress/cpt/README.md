# cpt/ — Qwen3 base CPT (Beaker)
Continued pretraining of Qwen3/Qwen3.5 bases on dolma3+longmino: dense vs landmark vs
sparse/compressive-landmark, lr sweeps, 8-node scale-ups; `_single_node` variants for smoke runs.
`interleaved/` = per-layer *attention-type* mixes (not data mixes); `debug/` = wikitext smoke
scripts; `launch_after_experiment.sh` chains a launch after a running experiment finishes.

## `interleaved/`
Two generations:

* **Qwen3-4B, 64k, `*-dolma3longmino.py`** — sparse-landmark interleaved with plain **full**
  attention (`alternating`, `Nsparse-1reg`, `first/second-half-full`), 4 nodes.
* **Qwen3.5-4B, 256k, `*-256k.py`** — sparse landmark interleaved with **regular** landmark on the
  8 full-attention layers of the hybrid (layers 3, 7, …, 31), on the longmino-512k mix, 2 nodes
  each. Four arms, all sharing `_qwen35_interleaved_landmark_256k_common.py`:
  `reg-first`, `reg-last`, `sparse-reg` (alternating, sparse first), `reg-sparse` (alternating,
  reg first). CP is capped at 4 by `SparseLandmarkAttention`, so DP = 4 on 2 nodes.
