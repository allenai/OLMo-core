# cpt/ — Qwen3 base CPT (Beaker)
Continued pretraining of Qwen3/Qwen3.5 bases on dolma3+longmino: dense vs landmark vs
sparse/compressive-landmark, lr sweeps, 8-node scale-ups; `_single_node` variants for smoke runs.
`interleaved/` = interleaved-data variants; `debug/` = wikitext smoke scripts;
`launch_after_experiment.sh` chains a launch after a running experiment finishes.
