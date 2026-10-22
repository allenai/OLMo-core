# olmo3_cptdrop — token-dropping CPT screen on OLMo 3 7B (2026-10-21)

Short trained screen from `allenai/Olmo-3-1025-7B@stage2-step47684` at 32k on the Dolma 3 Longmino pool:
dense vs length curriculum vs input-level token drop, scored by full-context dev CE at matched FLOPs.
Plan: `records/olmo3-cptdrop-screen-plan.md`. State: `ops/experiments/olmo3_cptdrop/STATUS.md`.

| file | role |
|---|---|
| `build_pool_data.py` / `build_data.sbatch` | longmino pool → `train32k` (one-document 32k windows, shuffled), `dev32k` (2^17+ buckets, never in the stage-3 mix), `short4k` (forgetting guard); `train_ctc_suite.py` shard layout + `rows.tsv` provenance |
| `convert_base.sbatch` | HF → OLMo-core model-only distcp for stage2 (CPT start) and stage3 (reference), with the converter's HF-parity validation |

Outputs stay out of git: `/data/prasann/olmo3_cptdrop/{data,base,logs}` on the build node, mirrored to
`s3://ai2-llm/checkpoints/prasanns/_transfer/olmo3_cptdrop/` for weka (Beaker) and Lambda node /tmp.
