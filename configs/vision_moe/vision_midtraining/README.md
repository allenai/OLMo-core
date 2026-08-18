# Vision midtraining

`Vision-Midtraining.py` model-only loads the permanent
`vision-alignment-joint-v1/step12000` treatment checkpoint, then trains the complete LM and
connector while keeping `vision.*` frozen and in eval mode. It uses native Stage-1 `document`
serialization, the six separate Stage-1 visual sources, and the official OLMo 0925 ingredient-1
NumpyFSL text mix; Tulu/SFT data is absent. Same-folder restarts require the exact saved run
contract before OLMo-core resumes full model, optimizer, trainer, and data-loader state.

The reviewed pilot defaults are 10.48576B tokens, 1,048,576-token global batches, sequence length
8192, LM LR `1e-5`, connector LR `2e-5`, and 50/50 target text/visual loss mass. It retains
Stage-1 `max_crops=8` and historical packing capacity `pack_max_crops=16`.

Measure all seven source means before a real run:

```bash
python src/scripts/train/Vision-Midtraining.py audit vision-midtraining-source-audit \
  --data.audit_output_path=/weka/oe-training-default/rustin/experiments/vision-moe/vision-midtraining/artifacts/source-means.json
```

Then review with ordinary OLMo-core overrides (replace `<SHA256>` with the receipt hash):

```bash
python src/scripts/train/Vision-Midtraining.py dry_run vision-midtraining-pilot \
  --data.mean_loss_weight_receipt=/weka/oe-training-default/rustin/experiments/vision-moe/vision-midtraining/artifacts/source-means.json \
  --data.mean_loss_weight_receipt_sha256=<SHA256>
```

Real `train` and `launch` commands fail closed without both receipt fields. A topology-only smoke
uses `--data.synthetic_smoke=true --max_tokens=1048576 --data.prefetch_workers=0`. The audit and
training contract assumes the configured Weka visual datasets are immutable; several raw PixMo
adapters do not currently expose independent content fingerprints.
