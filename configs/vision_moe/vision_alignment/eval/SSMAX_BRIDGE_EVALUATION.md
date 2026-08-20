# Paired SSMax bridge evidence

This path evaluates the two 1.4B Cx8 bridge runs without substituting checkpoint paths. Each arm
first produces one immutable manifest that binds its parent, training profile, recipe, validation
population, fixed matched-wrong rows, fixed attention probe, and the complete bytes of steps
0/100/200/250/300/400/500. Every later command accepts the manifest plus a raw-byte SHA pin and a
step from that closed set.

The finalizer accepts only the canonical per-arm specification at the saved clean Git revision and
records its repository-relative path, raw SHA-256, and Git blob identity. The 2x8 topology,
512 examples/source, seeds, 10,000 bootstrap samples, response windows, and every gate margin are
also hard-coded protocol constants; an ad-hoc weaker but structurally valid specification is not
accepted.

The parent binding includes the full native pretraining checkpoint identity: every DCP file and
all 64 trainer-rank states, in addition to the config, marker, DCP metadata, and logical model
inventory. Fresh bridge startup recomputes this byte identity on rank 0 before loading any parent
tensor and records it in the immutable parent-load receipt.

The per-arm specifications are:

- `ssmax_head_qknorm_bridge_manifest_v1.json`
- `ssmax_no_qknorm_bridge_manifest_v1.json`

Run the QK manifest materializer first. It creates the shared caption/transcript pairing files
once. The no-QK invocation then validates and pins the same canonical pairing bytes. Do not run the
two materializers concurrently.

```bash
PYTHONPATH=src python src/scripts/eval/vision_alignment_ssmax_bridge_manifest.py \
  --spec configs/vision_moe/vision_alignment/eval/ssmax_head_qknorm_bridge_manifest_v1.json \
  --output /weka/oe-training-default/rustin/experiments/vision-ssmax-molmofication/vision-alignment/evidence/bridge-v1/ssmax_head_qknorm/manifest.json

PYTHONPATH=src python src/scripts/eval/vision_alignment_ssmax_bridge_manifest.py \
  --spec configs/vision_moe/vision_alignment/eval/ssmax_no_qknorm_bridge_manifest_v1.json \
  --output /weka/oe-training-default/rustin/experiments/vision-ssmax-molmofication/vision-alignment/evidence/bridge-v1/ssmax_no_qknorm/manifest.json
```

Record `sha256sum` for each finalized manifest. For every retained step, launch the GPU evaluator
through the fixed 2x8 evidence launcher. It pins `ai2/scaling-ladders`, Holmes, urgent priority,
an eight-hour minimum runtime, the shared Weka mount, and a clean immutable source checkout:

```bash
PYTHONPATH=src python src/scripts/beaker_launch_vision_ssmax_evidence.py launch bridge \
  ssmax-ARM-bridge-stepSTEP -- \
  --manifest /weka/oe-training-default/rustin/experiments/vision-ssmax-molmofication/vision-alignment/evidence/bridge-v1/ARM/manifest.json \
  --expected-manifest-sha256 MANIFEST_SHA256 \
  --step STEP \
  --work-dir /weka/oe-training-default/rustin/experiments/vision-ssmax-molmofication/vision-alignment/evidence/bridge-v1/ARM/stepSTEP/work \
  --output /weka/oe-training-default/rustin/experiments/vision-ssmax-molmofication/vision-alignment/evidence/bridge-v1/ARM/stepSTEP/matched-state.json
```

One invocation emits all model-dependent evidence for that step:

- exact rehash and strict generic `MultimodalLM` DCP inventory/load;
- fixed correct versus exact-geometry-wrong rows for captions and transcripts;
- per-example and aggregate first-8/first-32/all response CE gaps, paired bootstrap intervals,
  and win rates;
- exact step-0 versus candidate hashes for every frozen LM/vision tensor and every non-image input
  embedding row;
- full connector, SigLIP, and six image-row descriptors; and
- the fixed 32-row Scalable-Softmax logit/magnitude/entropy probe. Step 0 is the correct
  parent-composite baseline: strict parent LM plus pinned SigLIP and deterministic new-component
  initialization, not a bare text-only parent mislabeled as multimodal.

The data cursor/loss-mass audit is CPU-only and also runs once per retained step:

```bash
PYTHONPATH=src python src/scripts/eval/vision_alignment_ssmax_bridge_health.py \
  --manifest MANIFEST.json \
  --expected-manifest-sha256 MANIFEST_SHA256 \
  --step STEP \
  --work-dir /weka/oe-training-default/rustin/experiments/vision-ssmax-molmofication/vision-alignment/evidence/bridge-v1/ARM/stepSTEP/health-work \
  --output /weka/oe-training-default/rustin/experiments/vision-ssmax-molmofication/vision-alignment/evidence/bridge-v1/ARM/stepSTEP/health.json
```

It reconstructs the pinned recipe and un-packed GatedDeltaNet-safe loader for all 16 ranks,
replays exactly `STEP` batches, requires exact equality with every saved loader cursor, and checks
zero data errors plus delivered 70/30 supervised-loss mass within two percentage points.

After raw-byte hashing every receipt, build one report per arm. Supply each of the seven options
once with `STEP` equal to 0, 100, 200, 250, 300, 400, or 500:

```bash
PYTHONPATH=src python src/scripts/eval/vision_alignment_ssmax_bridge_promotion.py promote \
  --manifest MANIFEST.json --expected-manifest-sha256 MANIFEST_SHA256 \
  --matched STEP=MATCHED_STATE.json --expected-matched-sha256 STEP=MATCHED_SHA256 \
  --health STEP=HEALTH.json --expected-health-sha256 STEP=HEALTH_SHA256 \
  ...repeat both receipt options for every retained step... \
  --output /weka/oe-training-default/rustin/experiments/vision-ssmax-molmofication/vision-alignment/evidence/bridge-v1/ARM/promotion.json
```

The bridge gate requires exact frozen state and healthy data at every saved point; treats step 0
as a descriptive composite baseline rather than claiming a null from a nonsignificant test;
positive gap lower bounds at steps 250/300/400/500; at least 80% of the
step-250 first-8/first-32 gap retained at step 500; no more than a 2% correct-image CE increase from
step 250 to 500; and a significantly improved paired correct-image CE from step 0 to 500. Attention
collapse flags are retained as triage evidence, not silently converted into a quality gate.

After a human reviews that deviation-free report, create the v4 parent gate explicitly. The
builder reopens the full report, every bound receipt, the manifest, and the live step-500
checkpoint. It accepts no waiver and never supplies an approver or timestamp default:

```bash
PYTHONPATH=src python src/scripts/eval/vision_alignment_ssmax_bridge_promotion.py approve \
  --report /weka/oe-training-default/rustin/experiments/vision-ssmax-molmofication/vision-alignment/evidence/bridge-v1/ARM/promotion.json \
  --expected-report-sha256 PROMOTION_SHA256 \
  --approved-by DURABLE_HUMAN_IDENTITY \
  --approved-at ISO8601_TIMESTAMP_WITH_TIMEZONE \
  --output /weka/oe-training-default/rustin/experiments/vision-ssmax-molmofication/vision-alignment/evidence/bridge-v1/ARM/parent-gate-v4.json
```

Finally compare the two full trajectories. This command first proves bit-identical step-0 SigLIP,
connector, and image-row state, then reports row-paired QK-minus-no-QK gap and correct-CE deltas at
every step/window. It keeps final capability separate from adaptation: the latter is the paired
step-0-normalized difference-in-differences for semantic gap growth and correct-image CE
improvement. It names an absolute or adaptation-dominant arm only when all six corresponding
step-500 intervals exclude zero in the same direction; otherwise that conclusion is explicitly
inconclusive. Comparison first reopens and exactly rebuilds both raw-SHA-pinned promotion reports;
both must be deviation-free passes. It also requires identical data contracts, dataset
fingerprints, initial/final/replayed loader cursors, source delivery, replay protocol, and equal
zero health counters at every step. A failed or data-incompatible arm cannot receive a ranking.

```bash
PYTHONPATH=src python src/scripts/eval/vision_alignment_ssmax_bridge_promotion.py compare \
  --left-promotion /weka/oe-training-default/rustin/experiments/vision-ssmax-molmofication/vision-alignment/evidence/bridge-v1/ssmax_head_qknorm/promotion.json \
  --expected-left-promotion-sha256 LEFT_SHA256 \
  --right-promotion /weka/oe-training-default/rustin/experiments/vision-ssmax-molmofication/vision-alignment/evidence/bridge-v1/ssmax_no_qknorm/promotion.json \
  --expected-right-promotion-sha256 RIGHT_SHA256 \
  --output /weka/oe-training-default/rustin/experiments/vision-ssmax-molmofication/vision-alignment/evidence/bridge-v1/paired-comparison.json
```

BLINK jigsaw, MathVista geometry, and open-ended academic evaluation are intentionally outside this
bridge-mechanics gate and should be attached at the later downstream evaluation stage.
