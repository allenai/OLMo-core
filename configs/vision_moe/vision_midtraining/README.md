# Vision midtraining

`Vision-Midtraining.py` model-only loads the permanent
`vision-alignment-joint-v1/step12000` treatment checkpoint, then trains the complete LM and
connector while keeping `vision.*` frozen and in eval mode. It uses native Stage-1 `document`
serialization, the six separate Stage-1 visual sources, and the historical OLMo 3 7B midtraining
source mixture. The exact source-list blob is restored at its original path and pinned to SHA-256
`15ee181c199bb89b118672737340153093ace8b5765fd87f760aa944669e2cff`; it includes the
curated web, science, code, math, reasoning, and instruction/SFT sources. Same-folder restarts
require the exact saved run contract before OLMo-core resumes full model, optimizer, trainer, and
data-loader state.

Every arm uses 2 Holmes nodes with 8 B300 GPUs each, a 10.48576B-token learning-rate horizon,
1,048,576-token global batches, sequence length 8192, LM LR `1e-5`, and connector LR `2e-5`.
It retains Stage-1 `max_crops=8` and historical packing capacity `pack_max_crops=16`. The only
mixture treatments are `v100`, `vt50` (default), `vt80`, and `vt90`, where the number is the
expected text supervised-loss-mass percentage. The remaining loss mass preserves the visual
source proportions caption/transcript/basic-points/high-frequency-points/count/CoSyn at
50/20/10/4/10/6.

Measure the six visual sources once for `v100`, and all seven sources once for the three text arms:

```bash
python src/scripts/train/Vision-Midtraining.py audit vision-midtraining-source-audit-v100-va12k \
  --data.mixture_arm=v100 \
  --data.audit_output_path=/weka/oe-training-default/rustin/experiments/vision-moe/vision-midtraining/artifacts/source-means-v100-va12k.json

python src/scripts/train/Vision-Midtraining.py audit vision-midtraining-source-audit-vt-va12k \
  --data.mixture_arm=vt50 \
  --data.audit_output_path=/weka/oe-training-default/rustin/experiments/vision-moe/vision-midtraining/artifacts/source-means-vt-va12k.json
```

Then review or launch an arm with ordinary OLMo-core overrides. For a matched 500-step pilot, set
`hard_stop_tokens=524288000` while leaving `max_tokens` unchanged; this stops early without
compressing the full-run cosine schedule. Replace the receipt placeholders with the corresponding
six- or seven-source receipt hash:

```bash
python src/scripts/train/Vision-Midtraining.py dry_run vision-midtraining-va12k-vt50-pilot500-v1 \
  --data.mixture_arm=vt50 \
  --hard_stop_tokens=524288000 \
  --checkpoint_interval=500 \
  --ephemeral_checkpoint_interval=null \
  --data.mean_loss_weight_receipt=/weka/oe-training-default/rustin/experiments/vision-moe/vision-midtraining/artifacts/source-means-vt-va12k.json \
  --data.mean_loss_weight_receipt_sha256=<SHA256>
```

Real `train` and `launch` commands fail closed without both receipt fields. A topology-only smoke
uses `--data.synthetic_smoke=true --max_tokens=1048576 --data.prefetch_workers=0`. The audit and
training contract assumes the configured Weka visual datasets are immutable; several raw PixMo
adapters do not currently expose independent content fingerprints. The vision-only `v100` arm and
synthetic smoke jobs launch without Google credentials. Real text-bearing arms require the
`GOOGLE_CREDENTIALS` Beaker secret in the pinned workspace; the recipe never creates or copies
that secret.
