# Migration to `akshitab/moe-v2-core`

## Provenance

- Source experiment branch: `jacobm/olmo-ddp` at `d7198425831a0f45eb483acce68296891bd86660`.
- Upstream migration base: `akshitab/moe-v2-core` at
  `f5376c18424e3f7329fa6e39312c63b84c5f845a`.
- Migration branch: `jacobm/moe-v2-core`.

This branch starts from upstream and imports Jacob's experiment layer. It does
not merge or copy the old branch's `src/olmo_core` implementation.

## Config boundary

`v2/moe_v2_core_adapter.py` is the only compatibility layer for recorded
`olmo-ddp` configs. It strictly translates:

1. identical `attention_norm` and `feed_forward_norm` configs to `layer_norm`;
2. `d_attn / n_heads` to `head_dim`;
3. flat legacy expert-parallel controls to `ExpertParallelConfig`; and
4. Muon-only optimizer controls only after requiring `use_muon=false`.

Checkpoint tensor names and values are not rewritten.

## Gates

- [x] Import tracked v1/v2 configs, launchers, results, plots, and docs.
- [x] Copy the ignored results cache so result refreshes do not redownload all W&B data.
- [x] Update active Beaker wrappers and manifests to this worktree.
- [x] Match the old branch's three model-summary outputs byte-for-byte.
- [x] Build all 20 active family/size/profile combinations with unchanged parameter counts.
- [x] Dry-run 275M EP1 and 1.2B EP8 pretraining configs.
- [x] Dry-run weight-only midtraining and 64K long-context continuation configs.
- [ ] Rerun the strict tensor/logit/block/router checkpoint gate on the pinned upstream tip.
- [ ] Run representative pretraining, EP8, midtraining, long-context, and eval smokes.
- [ ] Run the 275M throughput matrix only after the functional gates pass.
- [ ] Make this branch canonical and mark `jacobm/olmo-ddp` read-only.

## Checkpoint policy

Existing OLMo DDP checkpoint directories remain the source of model weights.
Pretraining-to-midtraining and midtraining-to-long-context transitions load
weights only and start a fresh optimizer. Cross-branch optimizer-state resume
is not part of this migration.

The strict verifier must require exact key/shape/value mapping and
`torch.equal` for fixed-input logits, block outputs, and router tensors. A
tolerance-based comparison is not an acceptable migration gate.
