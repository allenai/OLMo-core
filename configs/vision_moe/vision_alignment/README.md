# Vision alignment continued pretraining

This directory owns the new vision-alignment recipe implemented by
`src/scripts/train/Vision-Alignment.py`. It is separate from the historical
`Molmo2-Stage1.py` and does not reuse that run's save folders, optimizer state, Tulu replay,
or W&B identity.

The recipe has three explicit phases:

1. `bridge`: bare s002 LM + pinned pristine SigLIP; train only the connector and six
   input-only image-token rows on fixed-prompt, document-formatted PixMo captions and
   transcripts.
2. `perception`: model-only fork from a bridge checkpoint; fresh optimizer/data cursor;
   train connector + vision on the audited perception mix while the LM remains frozen.
3. `joint`: model-only fork from a perception checkpoint; fresh optimizer/data cursor;
   train connector + vision + the LM blocks/norms/routers at differential learning rates, with
   exact native `OLMo-mix-0925` replay. Ordinary lexical input-embedding rows and the untied
   output projection remain frozen; only the six image-token input rows adapt.

Changing phase, mixture, loss weighting, trainability, or learning-rate policy is never an
exact resume. Each such change requires a new run name and save folder. An interruption within
one unchanged phase resumes from that phase's save folder with full optimizer, trainer, RNG,
and mixture-loader state.

Mixture targets are effective supervised-loss mass, not example probabilities. Before real
training, audit the exact serialized sources and populate
`data.mixture.mean_loss_weight.<source>` from the canonical audit JSON. The loader derives
example sampling probabilities as `target_mass / mean_loss_weight` and checkpoints source
content fingerprints when the source advertises one. Online metrics report delivered examples,
tokens, supervised tokens, summed loss weight, realized loss-mass share, and target error for
every source. A bridge with missing calibration refuses to construct its loader.

The implementation is split by responsibility:

- `src/scripts/train/Vision-Alignment.py`: phase composition, immutable lineage checks,
  train/eval construction, and Holmes launch entry point.
- `src/olmo_core/data/multimodal/mixtures/vision_alignment.py`: checked-in effective-loss
  targets and calibration math.
- `src/olmo_core/data/multimodal/vision_alignment_sources.py`: the sole importable visual
  source/preprocessing registry plus deterministic probe identities and model-input hashing.
- `src/olmo_core/data/multimodal/native_text_replay.py`: bounded, map-style native-token
  replay with no detokenization, prompts, or chat tokens.
- `src/scripts/data/export_vision_alignment_probe.py`: canonical exact-runtime probe exporter.
- `src/scripts/data/audit_vision_alignment_mix.py`: strict probe audit and loss-mass calibration.
- `src/scripts/data/build_s002_replay_manifest.py`: deterministic, disjoint native replay and
  holdout manifest builder.

The s002 replay logistics catalog is deliberately limited to
`{parent_path_index, local_materialized_path}` and must map every row of the exact expanded
parent path list. Labels, remote paths, source IDs, token counts, and byte digests may not be
supplied by that catalog. They derive from the pinned `OLMo-mix-0925` and checkpoint path
manifests plus a separately reviewed upstream per-object byte inventory. No authoritative
inventory currently exists in this checkout, so its path/SHA constants remain unset and joint
production fails closed until the data owners publish and review it.

After that provenance artifact is pinned, the builder streams each local materialized token
file once, verifies its authoritative size and SHA-256, and emits
`native-text-replay-verification.json`. Production joint profiles pin that receipt's exact
SHA-256; runtime validates the receipt and current file sizes without re-hashing terabytes of
replay data. Train and holdout manifests are deterministically disjoint, and the holdout is an
intrinsic native-LM evaluator. Validate the post-provenance build without writing artifacts:

```bash
PYTHONPATH=src python src/scripts/data/build_s002_replay_manifest.py \
  --catalog /path/to/s002-replay-sources.json \
  --output-dir /path/to/vision-alignment/replay \
  --sequence-length 8192 \
  --train-tokens 819100000 \
  --holdout-tokens 8191000 \
  --dry-run
```

After reviewing that output, rerun the same command without `--dry-run` to write the manifests
and verification receipt.

First export exact `dataset.get(index, epoch=0)` rows through the same pinned tokenizer and
source registry used by training. The image-hash input is the sorted, unique SHA-256 inventory
for all training images:

```bash
PYTHONPATH=src python src/scripts/data/export_vision_alignment_probe.py \
  --phase bridge \
  --dataset-path /path/to/pixmo-cap \
  --image-hashes /path/to/train-images.sha256 \
  --output-dir /path/to/vision-alignment/bridge-probe
```

Then audit the generated catalog before copying its `mean_loss_weight` values and artifact
fingerprint into a production profile:

```bash
PYTHONPATH=src python src/scripts/data/audit_vision_alignment_mix.py \
  /path/to/vision-alignment/bridge-probe/vision-alignment-source-catalog.json \
  --phase bridge \
  --output /path/to/bridge-source-audit.json
```

The canonical probe covers at least 1,024 deterministic rows per source and pins the full live
dataset fingerprint, row count, selected indices, and hashes of every model-consumed serialized
array. Training rejects legacy or caller-assembled catalogs, verifies the exact registry,
exporter, auditor, tokenizer, and preprocessing contract, and recomputes all pinned row hashes
from the newly built live dataset before wrapping it or constructing a loader. Transcript
sources also run their dataset-wide required-annotation scan before probe export, training, and
validation.

`bridge/synthetic_smoke.yaml` is only a code/topology smoke profile. Its calibration was
measured over the deterministic synthetic dataset and is not valid for PixMoCap. A materialized
real-data profile is accepted only after its row manifest, held-out image-hash split, and
loss-mass audit are durable. `bridge/real_canary_v1.yaml.template` remains the deliberately
non-launchable template for reconstructing the first 250-step real-data canary; every
`__PIN_*__` value must be replaced with an exact reviewed artifact before use.
Production evaluation is mandatory at startup and finish and uses a separately pinned validation
manifest whose image hashes have zero overlap with training.
The validation-manifest v3 schema points to the actual sorted, unique train and validation
image-SHA256 files and pins their byte digests and row counts. Startup reads both files and
computes the set intersection; a caller-supplied overlap count is not trusted. The live Arrow
split fingerprint and length must also match the manifest.

The synthetic smoke runs one step during the connector warmup. At step 1 the scheduler applies
only 1% of the peak connector LR, so the run includes one tiny update: it proves config,
topology, forward/backward, optimizer construction, and checkpoint plumbing, but it is not a
meaningful learning or quality test. Its checkpoint is forbidden as a parent of another phase.

Each in-training visual intrinsic suite reports ordinary held-out response CE/PPL and a paired
blank-image CE/PPL on the same deterministic examples. The blank control replaces normalized
image patches with zeros while preserving crop geometry, prompts, and labels. Its gap measures
sensitivity to this out-of-distribution image-path intervention; a positive gap is not evidence
of semantic image-response binding by itself. Use the separately versioned matched wrong-image
evaluator before promoting a checkpoint. It replaces only the image tensor with a distinct-content
validation image whose tensor shape and exact `pooled_patches_idx` array match the recipient, and
reports results on that explicitly pinned matched-eligible subset. Evaluation uses the phase's
supported per-rank instance capacity (four at 2,560 tokens, one at 8,192). Native replay holdout
uses a deterministic permutation so a bounded evaluation is not biased toward the first source
files in the manifest.

Inspect the synthetic profile without submitting:

```bash
PYTHONPATH=src python src/scripts/train/Vision-Alignment.py dry_run \
  vision-alignment-bridge-synthetic-smoke \
  --profile=configs/vision_moe/vision_alignment/bridge/synthetic_smoke.yaml
```

Profiles select only `ai2/holmes`; they never name individual nodes. Replacing `dry_run` with
`launch` still requires explicit launch approval.

## First real bridge canary

Do not point the canary at the unfiltered shared PixMoCap root. After committing the exact
builder bytes, run one builder process to hash the actual image bytes, preserve validation, and
remove every training row whose image content occurs in validation:

```bash
PYTHONPATH=src python src/scripts/data/build_vision_alignment_pixmo_cap.py \
  --output-dir=/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/artifacts/pixmo-cap-content-disjoint-v1 \
  --workers=32 \
  --scan-batch-size=4096 \
  --max-shard-size=2GB
```

The CLI defaults pin the canonical PixMoCap source path, split fingerprints, and row counts.
The final output is immutable and published by an atomic rename from the sibling
`.pixmo-cap-content-disjoint-v1.building` directory. If the process is interrupted, rerun the
identical command with `--resume`; never run multiple builders or resumers concurrently. The
DatasetDict path used by probe export and `data.pixmo_cap_path` is the artifact's `dataset/`
subdirectory, not the artifact root. The source audit and validation manifest must both bind the
resulting train-image inventory bytes.
The validation-manifest v3 also pins the filtered Arrow root, both live split fingerprints and
row counts, and the builder's exact SHA-256.

After exporting and auditing the filtered root with the commands above, copy the template to
`bridge/real_canary_v1.yaml` and replace every placeholder with:

- the absolute filtered Arrow root;
- the absolute successful bridge source-audit JSON and its `fingerprint` field;
- the audit's exact `mean_loss_weight.pixmo_caption` and
  `mean_loss_weight.pixmo_transcript` values; and
- the absolute builder-produced validation-manifest v3 JSON and SHA-256 of its exact bytes.

Do not replace the audited mean loss weights with the 70/30 targets. Selecting `phase: bridge`
locks the desired effective supervised-loss mass to 70% captions and 30% transcripts; the two
audited means convert those targets into example-sampling probabilities. The phase contract also
locks the connector LR to `2e-4` with a 100-step warmup and keeps the vision encoder and LM frozen.
These structural settings are intentionally not profile overrides.

The canary evaluates the ordinary and blank-image caption/transcript suites at startup, steps
100 and 200, and finish at step 250. The checkpointer writes permanent checkpoints at steps 0,
100 and 200; its normal post-train hook writes the final permanent step-250 checkpoint. Periodic
and ephemeral intervals are disabled, and `max_checkpoints=4` retains all four canary states.

The completed canary passed its technical and semantic gates. Under the exact matched wrong-image
protocol, the all-token caption gap and win rate progressed from `-0.0027 / 50.6%` at step 0 to
`+0.0420 / 95.1%` at step 250; transcript progressed from `+0.0007 / 51.2%` to
`+0.0329 / 92.6%`. Both gaps had confidence intervals containing zero at initialization, became
significantly positive by step 100, and improved monotonically through step 250. This supersedes
the earlier interpretation of the blank-image gap: blank images remain a path-ablation diagnostic,
not the semantic promotion gate.

Inspect the fully materialized profile without submitting:

```bash
PYTHONPATH=src python src/scripts/train/Vision-Alignment.py dry_run \
  vision-alignment-bridge-real-canary-v1 \
  --profile=configs/vision_moe/vision_alignment/bridge/real_canary_v1.yaml
```

The step-250 checkpoint is only a gate candidate. Compare the startup/100/200/250 correct-image
and blank-image CE/PPL, then replay every saved checkpoint with the versioned matched wrong-image
evaluator using the same pinned pairing files. Verify finite optimization and router metrics,
inspect the delivered 70/30 loss-mass telemetry, and run the agreed language/numerical regression
checks. A human must then issue the pinned approved-parent gate; the canary does not approve
itself. Because duration is part of the trainable contract, do not extend this run in place. If a
longer bridge is needed, start a fresh named bridge from the bare checkpoint with a separately
reviewed profile.

Run the post-hoc binding check on one 8-GPU node. The completed canary receipts already produced
the immutable caption and transcript pairings, so every v3 replay pins their exact SHA-256 values
and writes to a new output path:

```bash
for step in 0 100 200 250; do
  PYTHONPATH=src torchrun --standalone --nproc-per-node=8 \
    src/scripts/eval/vision_alignment_matched_wrong.py \
    --checkpoint=/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/checkpoints/vision-alignment-bridge-real-canary-v1/step${step} \
    --pairing-dir=/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/evals/bridge-real-canary-v1-matched-wrong-v2/pairings \
    --expected-pairing-sha256=pixmo_caption=9d37a3719b51804c26214625b4651faee2046e1c2cdb21a8990add17230cdb31 \
    --expected-pairing-sha256=pixmo_transcript=49d8b3f1b3b1e96a5547c1408750b1569668d1dfda7b57eeea1f33995908731a \
    --output=/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/evals/bridge-real-canary-v1-matched-wrong-v3/step${step}.json
done
```

Existing result paths fail closed; use a new path rather than `--overwrite-output` for scientific
receipts. The v3 evaluator hashes every checkpoint state file before model construction and
requires every model parameter and persistent buffer to have one unambiguous native load source
on every rank.

This metric is explicitly conditional on the exact-geometry matched-eligible subset recorded in
the pairing artifact. Compare `wrong_ce - correct_ce`, its bootstrap interval, and win rate across
the four checkpoints; do not compare it to the in-training blank-image gap as if they were the
same intervention or population.

## Full bridge refinement

`bridge/real_bridge_v1.yaml` is a fresh 500-step run from the bare checkpoint, not a resume from
the canary. Its connector and image-token-row scheduler exactly reproduces the successful canary:
100 warmup steps, cosine decay to `2e-5` at step 250, then the same floor through step 500. This
explicit `t_max=250` is important—a duration-derived 500- or 1,000-step cosine would keep the
connector near peak LR at step 250 and would be a different high-LR experiment.

The profile keeps the audited data and 70/30 effective-loss mix unchanged. It writes permanent
checkpoints at steps 0, 100, 200, 250, 300, 400, and 500, with recoverable ephemeral checkpoints
between them. Ordinary/blank intrinsic evaluation runs every 100 steps and at finish; semantic
selection uses the standalone matched wrong-image evaluator on every permanent checkpoint.

Validate the exact profile before launch:

```bash
PYTHONPATH=src python src/scripts/train/Vision-Alignment.py dry_run \
  vision-alignment-bridge-real-v1 \
  --profile=configs/vision_moe/vision_alignment/bridge/real_bridge_v1.yaml
```

After committing and pushing the exact validated revision, replace `dry_run` with `launch` to
submit the profile. Evaluate each committed checkpoint with the same pairing directory and the
two exact SHA-256 pins above, writing results under a new
`bridge-real-v1-matched-wrong-v3/` directory.

At step 250, require every caption/transcript `first_8`, `first_32`, and all-token gap lower
confidence bound to be positive. Require at least 90% of the canary's step-250 point gaps and
correct-image all-token CE no more than 2% above the canary. At steps 300/400/500, retain at least
80% of this run's step-250 `first_8` and `first_32` gaps, keep correct-image CE within the same 2%
bound, deliver caption/transcript loss mass within two percentage points of 70/30, and preserve
every frozen LM/vision tensor and non-image embedding row exactly. Do not use transcript
`first_1` as a gate because its generic opening token is not expected to identify image content.

The perception and joint directories contain no launch profile yet. Their default contracts
deliberately fail closed until explicit scalar-count, OCR/document, audited-alignment, and
native-replay artifacts are implemented and pinned.
