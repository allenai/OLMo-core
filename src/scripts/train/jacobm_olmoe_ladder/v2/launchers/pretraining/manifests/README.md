# Intervention manifests

Each YAML file is the source of truth for one pretraining intervention. Copy an
existing manifest, then update all of the following before rendering:

- semantic experiment key, run prefix, and checkpoint root;
- tested training wrapper and image;
- environment-variable names expected by that wrapper;
- exact sequence length, world size, and EP size;
- per-Cx optimizer batch, rank microbatch, and LR list;
- workspace, cluster, priority, mounts, and secrets.

LRs must be quoted scientific-notation strings so names remain stable (`1.6e-3`
becomes `lr1p6e-3`). The launcher rejects duplicate run names, invalid batch
factorizations, multi-GPU/world-size mismatches, EP other than one, missing
wrappers, and duplicate environment variables.

Do not add Beaker/W&B IDs to a manifest. A manifest describes intended work;
[`../../../RUNS.md`](../../../RUNS.md) records what actually launched, and
`plot_pretraining_wave.py` registers the exact W&B runs used for results.
