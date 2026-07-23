# V1 migration record

Cutover date: 2026-07-14.

## Source boundary

- Legacy repository: `/weka/oe-adapt-default/jacobm/olmoe3/OLMo-core`
- Legacy branch: `jacobm/olmoe-dev-v2`
- Recorded source commit: `a8bdb7613946dba70227c682fd90746ff3be4a3e`
- DDP destination branch: `jacobm/olmo-ddp`
- W&B project retained across the cutover: `ai2-llm/jacobm-olmoe-ladder`

The legacy working tree had uncommitted operational updates at cutover. This snapshot intentionally copies the working-tree versions of every tracked ladder file, so it includes those updates rather than silently reverting to the recorded commit. The modified ladder files were `EVAL_PLAN.md`, `experiments/midtraining/launch_275m_lr_search.sh`, `experiments/qwen3_like/qwen3_like_ladder.py`, `moe_a0_ladder.py`, and `tiny_275m.py`. A separate legacy change to `src/olmo_core/model_ladder/base.py` was not copied over the DDP implementation; it only enabled speed monitoring in the old trainer and is recorded here for provenance.

The snapshot contains all 295 tracked ladder files plus ten untracked but used operational files: seven eval target lists, `experiments/integration/launch_1p2b_baseline_lr.sh`, `launch_cx8_olmobase_evals.py`, and `launch_oe_eval_auto_vllm_smoke.sh`. Generated result-summary caches are present locally under `results/cache/` but ignored by Git. W&B history caches are merged into the DDP repository's ignored `.cache/` tree as documented in `ARTIFACTS.md`.

## Experiment completeness

The historical pretraining, midtraining, and eval result set was refreshed before migration. The eight absent pretraining cells are intentional: the 1.2B Cx1/Cx2/Cx4/Cx8 runs for both `high_total_96e_top4` and `huge_total_192e_top4` were skipped after smaller-scale sparsity results were sufficiently clear. They are not missing work.

The publication manifest therefore contains 224 model checkpoints: 200 final pretrained checkpoints and 24 final midtrained checkpoints. It selects only the canonical optimal-LR endpoint for each cell, not every LR-sweep checkpoint. Converted checkpoints contain model weights only; optimizer, scheduler, dataloader, and trainer state are excluded.

## Cutover policy

- The files in this `v1/` directory are the authoritative historical package for the legacy ladder.
- Legacy launchers are reference-only because the DDP training API differs.
- New DDP launchers and tracking code will live beside this package after smoke testing.
- Old checkpoint directories are not deleted or modified by migration tooling.
- Long-context and hybrid/GDN experiments start only after the migration is complete and require separate launch approval.
