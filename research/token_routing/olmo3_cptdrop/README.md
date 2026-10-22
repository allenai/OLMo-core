# olmo3_cptdrop — token-dropping CPT screen on OLMo 3 7B (2026-10-21)

Short trained screen from `allenai/Olmo-3-1025-7B@stage2-step47684` at 32k on the Dolma 3 Longmino pool:
dense vs length curriculum vs input-level token drop, scored by full-context dev CE at matched FLOPs.
Plan: `records/olmo3-cptdrop-screen-plan.md`. State: `ops/experiments/olmo3_cptdrop/STATUS.md`.

| file | role |
|---|---|
| `build_pool_data.py` / `build_data.sbatch` | longmino pool → `train32k` (one-document 32k windows, shuffled), `dev32k` (2^17+ buckets, never in the stage-3 mix), `short4k` (forgetting guard); `train_ctc_suite.py` shard layout + `rows.tsv` provenance |
| `convert_base.sbatch` | HF → OLMo-core model-only distcp for stage2 (CPT start) and stage3 (reference), with the converter's HF-parity validation |
| `measure_keep_rules.py` | CPU: kept fraction (real keep code, synthetic 32k rows) and FLOP-meter actual/dense ratio (OLMo 3 7B formula) of every drop arm in `run_arm.sbatch` |

Follow-up arms in `run_arm.sbatch` (2026-10-22, not part of round 1): `IFL_R4K` (`--st-keep-recent 4096`:
last 4,096 tokens kept whole, I-fl before), `IFL_RAMP` (`--st-keep-ramp 0.1,0.5`: first_last fraction
ramps 0.1 → 0.5 over the row), and token exit `IFL_X4` / `IFL_X8` / `IRAND_X4` / `IRAND_X8`
(`--tokexit-layer L`: the same keep rule, dropped tokens run blocks [0, L) on the full row and leave after
the first / second full-attention layer; `olmo_core/nn/token_exit.py`, tests `src/test/nn/token_exit_test.py`).

Outputs stay out of git: `/data/prasann/olmo3_cptdrop/{data,base,logs}` on the build node, mirrored to
`s3://ai2-llm/checkpoints/prasanns/_transfer/olmo3_cptdrop/` for weka (Beaker) and Lambda node /tmp.
