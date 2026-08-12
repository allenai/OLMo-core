# Bridge step500 promotion evidence

This directory documents the fail-closed promotion boundary for
`vision-alignment-bridge-real-v1/step500`. Do not launch perception until all receipts below
exist, the promotion bundle audits successfully, and a human has explicitly approved both
locked deviations in a v2 parent gate.

All output paths are immutable. If a command finds an existing output, inspect and pin that
artifact; do not use an overwrite flag. The commands assume the repository revision whose
`src/scripts/train/Vision-Alignment.py` SHA-256 is
`b8a96d946224e42cd0cb6422d27081da09265ea4d0e963f8e7509ac6f39267a5`.

## 1. Pinned image-free text sentinel (credentialed CPU)

This reads 128 deterministic first windows from the bare parent's exact expanded pretraining
manifest. It needs credentials that can range-read the listed S3 shards; the local login node
currently receives `AccessDenied`.

```bash
PYTHONPATH=src python src/scripts/eval/vision_alignment_promotion.py text-sentinel \
  --parent-checkpoint=/weka/oe-training-default/robertb/s002-step125500 \
  --parent-checkpoint-config-sha256=35ce23db053dd2204bc37783546f1b2f98eafb742488903773dd0ef3e5741146 \
  --parent-data-paths=/weka/oe-training-default/robertb/s002-step125500/data_paths.txt \
  --expected-parent-data-paths-sha256=f1155957f4f249fc17e1c7067512e7d881ce6675c6b854d5ce089c649cec1c2d \
  --sequence-length=256 \
  --examples=128 \
  --output=/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/evals/bridge-real-v1-promotion-v1/text-sentinel.json
```

Record the raw SHA-256 printed by the command as `TEXT_SENTINEL_SHA256`.

## 2. Frozen-state and text-retention receipts (one native EP8 GPU node)

This hashes both DCPs before model construction, proves native load completeness, loads bridge
step0 and step500 under the same EP8 backend, compares all 806 frozen tensors plus every
non-image input-embedding row, and evaluates every token in the pinned image-free sentinel.

```bash
PYTHONPATH=src torchrun --standalone --nproc-per-node=8 \
  src/scripts/eval/vision_alignment_state_text.py \
  --reference-checkpoint=/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/checkpoints/vision-alignment-bridge-real-v1/step0 \
  --checkpoint=/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/checkpoints/vision-alignment-bridge-real-v1/step500 \
  --matched-step500=/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/evals/bridge-real-v1-matched-wrong-v3/step500.json \
  --expected-matched-step500-sha256=28e4f9b5122250bd851781a879c75c52b67d6b578760afc21ac1f5d665c4430c \
  --text-sentinel=/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/evals/bridge-real-v1-promotion-v1/text-sentinel.json \
  --expected-text-sentinel-sha256=TEXT_SENTINEL_SHA256 \
  --frozen-output=/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/evals/bridge-real-v1-promotion-v1/frozen-state.json \
  --text-output=/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/evals/bridge-real-v1-promotion-v1/text-retention.json \
  --text-batch-size=4 \
  --checkpoint-load-threads=8 \
  --checkpoint-hash-workers=8
```

Both receipts must report `status: passed`. The text receipt requires finite per-token CE,
maximum absolute and relative CE deltas at most `1e-6`, and identical argmax tokens.

## 3. Exact cumulative loader replay (CPU)

This reconstructs the two audited production datasets and sequentially replays all 500 batches
for each of the 16 saved DP ranks. Every resulting version-5 packing cursor must equal the
corresponding `train/rank*.pt` state. This is an exhaustive image-preprocessing job and can take
hours; give it a credentialed, high-CPU worker and durable logs.

```bash
PYTHONPATH=src python src/scripts/eval/vision_alignment_loss_mass.py \
  --checkpoint=/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/checkpoints/vision-alignment-bridge-real-v1/step500 \
  --matched-step500=/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/evals/bridge-real-v1-matched-wrong-v3/step500.json \
  --expected-matched-step500-sha256=28e4f9b5122250bd851781a879c75c52b67d6b578760afc21ac1f5d665c4430c \
  --recipe=src/scripts/train/Vision-Alignment.py \
  --expected-recipe-sha256=b8a96d946224e42cd0cb6422d27081da09265ea4d0e963f8e7509ac6f39267a5 \
  --work-dir=/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/evals/bridge-real-v1-promotion-v1/loss-mass-work \
  --output=/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/evals/bridge-real-v1-promotion-v1/cumulative-loss-mass.json
```

The raw and active supervised-loss-mass shares must each be within two percentage points of
caption/transcript `70/30`, with zero data errors and exact final cursor equality.

## 4. Independent disjoint matched-pair replication (EP8)

The independent seed is the primary seed plus the locked `1,000,003` offset. Both recipients
and donors from the primary population are excluded. The remaining exact-geometry population
has 1,140 selectable rows per source, so a new 512-pair sample is feasible.

First run step0, which creates the new pairings:

```bash
PYTHONPATH=src torchrun --standalone --nproc-per-node=8 \
  src/scripts/eval/vision_alignment_matched_wrong.py \
  --checkpoint=/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/checkpoints/vision-alignment-bridge-real-v1/step0 \
  --pairing-dir=/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/evals/bridge-real-v1-independent-matched-wrong-v3/pairings \
  --exclude-pairing=pixmo_caption=/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/evals/bridge-real-canary-v1-matched-wrong-v2/pairings/pixmo_caption.json \
  --exclude-pairing=pixmo_transcript=/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/evals/bridge-real-canary-v1-matched-wrong-v2/pairings/pixmo_transcript.json \
  --expected-exclude-pairing-sha256=pixmo_caption=9d37a3719b51804c26214625b4651faee2046e1c2cdb21a8990add17230cdb31 \
  --expected-exclude-pairing-sha256=pixmo_transcript=49d8b3f1b3b1e96a5547c1408750b1569668d1dfda7b57eeea1f33995908731a \
  --pairing-seed=1006201 \
  --bootstrap-seed=2006204 \
  --work-dir=/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/evals/bridge-real-v1-independent-matched-wrong-v3/work-step0 \
  --output=/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/evals/bridge-real-v1-independent-matched-wrong-v3/step0.json
```

Record `.pairings.pixmo_caption.sha256` and `.pairings.pixmo_transcript.sha256` from step0 as
`INDEPENDENT_CAPTION_SHA256` and `INDEPENDENT_TRANSCRIPT_SHA256`. Then reuse and pin those exact
pairings for step500 while re-validating the primary exclusions:

```bash
PYTHONPATH=src torchrun --standalone --nproc-per-node=8 \
  src/scripts/eval/vision_alignment_matched_wrong.py \
  --checkpoint=/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/checkpoints/vision-alignment-bridge-real-v1/step500 \
  --pairing-dir=/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/evals/bridge-real-v1-independent-matched-wrong-v3/pairings \
  --expected-pairing-sha256=pixmo_caption=INDEPENDENT_CAPTION_SHA256 \
  --expected-pairing-sha256=pixmo_transcript=INDEPENDENT_TRANSCRIPT_SHA256 \
  --exclude-pairing=pixmo_caption=/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/evals/bridge-real-canary-v1-matched-wrong-v2/pairings/pixmo_caption.json \
  --exclude-pairing=pixmo_transcript=/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/evals/bridge-real-canary-v1-matched-wrong-v2/pairings/pixmo_transcript.json \
  --expected-exclude-pairing-sha256=pixmo_caption=9d37a3719b51804c26214625b4651faee2046e1c2cdb21a8990add17230cdb31 \
  --expected-exclude-pairing-sha256=pixmo_transcript=49d8b3f1b3b1e96a5547c1408750b1569668d1dfda7b57eeea1f33995908731a \
  --pairing-seed=1006201 \
  --bootstrap-seed=2006204 \
  --work-dir=/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/evals/bridge-real-v1-independent-matched-wrong-v3/work-step500 \
  --output=/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/evals/bridge-real-v1-independent-matched-wrong-v3/step500.json
```

Step0 must reproduce the null for every `first_8`, `first_32`, and `all` confidence interval.
Step500 must have positive lower bounds for all six windows, retain at least 80% of each primary
step500 `first_8` and `first_32` gap, and keep correct-image CE within +2% of primary step500.

## 5. Existing run-health receipt

The immutable optimizer/run-health receipt already exists at
`evals/bridge-real-v1-promotion-v1/optimizer-guard.json`, raw SHA-256
`251ac67e8c82bc15e6248264e2a28ed51fa037534f65da73830bf7a14197d653`. It reopens all 16 trainer
rank states, all seven permanent checkpoint markers, and the raw W&B output log. It records the
single step356 safety-guard skip and no other anomaly.

## 6. Build, re-audit, and approve

After every receipt passes, build the immutable bundle:

```bash
PYTHONPATH=src python src/scripts/eval/vision_alignment_promotion.py build \
  --checkpoint=/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/checkpoints/vision-alignment-bridge-real-v1/step500 \
  --frozen-state=/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/evals/bridge-real-v1-promotion-v1/frozen-state.json \
  --text-retention=/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/evals/bridge-real-v1-promotion-v1/text-retention.json \
  --cumulative-loss-mass=/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/evals/bridge-real-v1-promotion-v1/cumulative-loss-mass.json \
  --optimizer-guard=/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/evals/bridge-real-v1-promotion-v1/optimizer-guard.json \
  --canary-step250=/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/evals/bridge-real-canary-v1-matched-wrong-v3/step250.json \
  --bridge-step250=/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/evals/bridge-real-v1-matched-wrong-v3/step250.json \
  --bridge-step500=/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/evals/bridge-real-v1-matched-wrong-v3/step500.json \
  --independent-step0=/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/evals/bridge-real-v1-independent-matched-wrong-v3/step0.json \
  --independent-step500=/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/evals/bridge-real-v1-independent-matched-wrong-v3/step500.json \
  --output=/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/evals/bridge-real-v1-promotion-v1/promotion-bundle.json
```

Record the printed raw SHA as `PROMOTION_BUNDLE_SHA256`, then independently re-audit it:

```bash
PYTHONPATH=src python src/scripts/eval/vision_alignment_promotion.py audit \
  --bundle=/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/evals/bridge-real-v1-promotion-v1/promotion-bundle.json \
  --expected-sha256=PROMOTION_BUNDLE_SHA256 \
  --expected-checkpoint=/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/checkpoints/vision-alignment-bridge-real-v1/step500 \
  --expected-checkpoint-config-sha256=41df40c299f4f3101c3ef58d657d99fb624194beaee7321ea456727212be1dad
```

Only the accountable human approver should run the next command. It requires both deviations to
be named explicitly; unknown, missing, duplicated, or free-form waivers fail closed.

```bash
PYTHONPATH=src python src/scripts/eval/vision_alignment_promotion.py approve \
  --bundle=/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/evals/bridge-real-v1-promotion-v1/promotion-bundle.json \
  --expected-sha256=PROMOTION_BUNDLE_SHA256 \
  --approved-by=HUMAN_IDENTITY \
  --approved-at=ISO_8601_UTC_TIMESTAMP \
  --approve-waiver=step250_caption_first32_90pct_canary \
  --approve-waiver=step356_optimizer_guard_skip \
  --output=/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/evals/bridge-real-v1-promotion-v1/parent-gate-v2.json
```

Production perception requires this v2 gate's exact path and raw SHA in
`initialization.parent_gate_path` and `initialization.parent_gate_sha256`. Legacy v1 gates remain
readable only for old/non-production test flows.
