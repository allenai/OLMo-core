# Jacob MoE Ladder Migration Plan

## Cutover implementation

The legacy experiment package has moved to
`src/scripts/train/jacobm_olmoe_ladder/v1/`. Its `MIGRATION.md` records the
source commit and dirty working-tree boundary, `DEFAULT_RUN_SETTINGS.md`
summarizes the inherited regime, and `ARTIFACTS.md` defines local and GCS
storage. The package includes all historical docs, ledgers, configs, launchers,
plots, generated results, and eval targets. Legacy launchers are reference-only.

W&B histories have been consolidated into the repository's ignored `.cache/`
tree, and result/download caches were copied under the v1 results directory.
Checkpoint publication, legacy HF export upload, and legacy eval-output upload
use restartable jobs with durable receipts; no old checkpoint directory is
automatically cleaned up.

Long-context and hybrid/GDN work is intentionally deferred until checkpoint and
artifact publication completes. Neither experiment class is launched as part of
the cutover.

## Architecture alignment experiments

After the current experiment wave, train each 275M wide-integration variant from scratch:

- **Control:** current wide integration recipe.
- **Hybrid:** replace sliding-attention layers with GatedDeltaNet while holding geometry, global-attention placement, RoPE, and initialization fixed.
- **Aligned geometry and mixer ratio:** use the dense ladder's 275M width, depth, attention geometry, and four-GDN/one-global pattern while retaining our MoE and dense-first-FFN design. The dense first FFN may still use GDN, preserving the exact 20% global-attention ratio.
- **NoPE:** on the hybrid recipe, remove RoPE only from global-attention layers; train from initialization.
- **Initialization:** on the control recipe, change only the initialization standard deviation from 0.01 to 0.02.

Combine neutral-to-positive interventions in one 275M pilot and confirm that they compose before running the full pretraining ladder, mid-training, and 8K-to-65K long-context stage. Compare both token-matched quality and active-parameter/FLOP efficiency because aligned geometry changes compute.

## Checkpoint migration rule

- Convert model weights only; do not retain optimizer or trainer state.
- Convert the final pretrained and final mid-trained checkpoint for every selected run.
- Cover all four model sizes and all four pretraining data multiples for the finalized main intervention and integration families.
- Run each conversion as an individual batch job with structural, load, and logits verification.
- Finalize the checkpoint manifest and storage estimate before launching conversion jobs.
