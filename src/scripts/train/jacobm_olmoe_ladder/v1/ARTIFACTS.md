# Artifact layout

## Converted DDP checkpoints

Canonical GCS root:

```text
gs://ai2-llm/checkpoints/jacobm/olmoe3/olmo-ddp-converted/v1/
  pretraining/<family>/<model_size>/cx<data_multiple>/
  midtraining/<family>/<model_size>/cx<data_multiple>/
```

Each checkpoint directory contains model-only distributed checkpoint shards, `config.json`, a provenance README, conversion reports, exhaustive exact-tensor verification, exact-logits verification, GCS integrity metadata, and a success marker written last. Source step numbers are recorded in metadata rather than encoded in the destination path.

Local converted checkpoints remain under `/weka/oe-adapt-default/jacobm/olmoe3/olmo-ddp-migration/converted-checkpoints/`. Migration tooling does not clean them up.

## Legacy HF exports and eval outputs

The preserved upload destinations are:

```text
gs://ai2-llm/checkpoints/jacobm/olmoe3/olmo-ddp-converted/v1/hf/legacy_exports/
gs://ai2-llm/checkpoints/jacobm/olmoe3/olmo-ddp-converted/v1/evals/legacy_outputs/
gs://ai2-llm/checkpoints/jacobm/olmoe3/olmo-ddp-converted/v1/manifests/
```

Their source trees are `/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/hf-checkpoints` and `/weka/oe-training-default/ai2-llm/evals/jacobm/olmoe3`. Uploads preserve the relative source tree so legacy references remain traceable.

## Analysis caches

- W&B histories: `<repo>/.cache/jacobm_olmoe_ladder/v1/wandb_histories/`
- W&B override: `OLMOE3_WANDB_CACHE_DIR`
- Result summaries and downloaded OLMo-base outputs: `results/cache/` within this package

Both cache locations are intentionally ignored by Git. They are migrated data, not source artifacts. The W&B directory includes `migration_report.json`, which records its source directories and selection counts.
