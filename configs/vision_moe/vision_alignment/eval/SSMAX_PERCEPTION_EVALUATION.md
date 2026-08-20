# SSMax perception causal-pair evidence

This is a new evidence boundary. The historical s002 perception module describes one completed
run and contains intentional run-specific constants and waivers; none of them authorize an SSMax
checkpoint.

SSMax recurrent blocks cannot consume the multi-annotation `subsegment_ids` representation.
After immutable provenance selection, the recipe deterministically selects one response branch
from `(source, stable raw index, epoch, data seed 95818)`. Training remains epoch-addressed;
validation/evidence materialize the backing row at epoch zero as well as selecting at epoch zero.
Every positive target is then reweighted by the exact single-response convention. The independent
wrong-image pairing/bootstrap seed remains 6198 and must never be substituted for the projection
seed.

Before any SSMax perception profile can be reviewed, build the immutable projected loss-mass and
held-out reconstruction receipt. The producer replays the source-audit train panel and every one
of the 512 provenance-selected validation rows per source, binds the raw+semantic audit and
provenance identities, hashes every projected row and branch decision, and publishes with atomic
no-replace semantics:

```bash
PYTHONPATH=src python src/scripts/data/build_ssmax_single_response_calibration.py \
  --phase perception \
  --selection-manifest /weka/.../vision-alignment-perception-provenance.json \
  --expected-selection-manifest-sha256 RAW_SHA256 \
  --source-audit /weka/.../perception-source-audit.json \
  --expected-source-audit-sha256 RAW_SHA256 \
  --expected-source-audit-fingerprint SEMANTIC_SHA256 \
  --hf-cache-dir /weka/.../hf-cache/hub \
  --output /weka/.../ssmax-single-response-calibration-v1/perception.json
```

Copy its exact per-source projected means and raw receipt SHA into both arm profiles. Runtime first
revalidates the ordinary serialized-source audit, then wraps those selected rows with the same
projection. Every rank checks the contract and canonical selected-dataset fingerprint; rank zero
alone reconstructs each full train and validation panel and broadcasts any failure, avoiding a
16-rank Weka replay storm. SSMax also sets both data-error limits to zero, so a row that makes the
run ineligible fails immediately. Historical s002 loader limits and serialization are unchanged.

For each SSMax model variant, create two reviewed profiles only after that variant's permanent
bridge step 500 has a human-approved v4 gate. Both profiles must use the same bridge parent,
visual provenance, source audit, seeds, two-node HSDP topology, duration, data/eval cadence, and
checkpoint cadence. The recipe derives the only causal intervention: the control adds `vision.*`
to `freeze_params` and changes the vision group LR from the treatment's positive value to zero.
Do not add a profile or allowlist entry containing a placeholder.

Profiles must permanently retain steps 0/500/1000/2000/3000/4000 so early-signal and sample-
efficiency trajectories remain observable. After both runs contain those checkpoints, copy
`ssmax_perception_pair_manifest_v1.json.template` to a concrete reviewed spec and replace every
placeholder. Finalization refuses partial runs, creates or verifies exact-geometry wrong-image
pairings for all eight provenance-selected validation sources, hashes every DCP and trainer-rank
file, verifies both checked-in profile blobs and both canonical evidence-producer blobs at the
saved clean git ref, and revalidates the common v4 bridge gate:

```bash
PYTHONPATH=src python src/scripts/eval/vision_alignment_ssmax_perception_manifest.py \
  --spec /path/to/reviewed-pair-spec.json \
  --output /immutable/evidence/pair-manifest.json
```

Run the GPU producer for every arm and step in `{0,3000,4000}` through the fixed 2x8 evidence
launcher. It pins the scaling-ladders workspace, Holmes, urgent priority, an eight-hour minimum
runtime, shared Weka, and a clean immutable source checkout. The producer accepts no arbitrary
checkpoint path. At startup both evaluation and health producers prove that `__file__` is the
canonical repo-relative source whose live bytes and saved Git blob equal the manifest pin; receipts
retain that durable source identity instead of an ephemeral Beaker clone path. Each receipt proves
strict generic model-only DCP coverage/load, hashes the full
logical model and frozen surfaces relative to the arm's step 0, evaluates correct versus
exact-geometry-wrong rows at first 1/8/32/all response tokens, and runs the immutable native-text
sentinel. It also reconstructs the manifest-bound PixMo-caption attention probe and captures
logit/QK magnitude, entropy, effective context, and routing in every step 0/3000/4000 receipt:

```bash
PYTHONPATH=src python src/scripts/beaker_launch_vision_ssmax_evidence.py launch perception \
  ssmax-VARIANT-perception-ARM-stepSTEP -- \
  --manifest /weka/oe-training-default/rustin/experiments/vision-ssmax-molmofication/vision-alignment/evidence/perception-v1/VARIANT/pair-manifest.json \
  --expected-manifest-sha256 SHA256 --arm treatment --step 4000 \
  --work-dir /weka/oe-training-default/rustin/experiments/vision-ssmax-molmofication/vision-alignment/evidence/perception-v1/VARIANT/work/ARM-stepSTEP \
  --output /weka/oe-training-default/rustin/experiments/vision-ssmax-molmofication/vision-alignment/evidence/perception-v1/VARIANT/ARM-stepSTEP-evaluation.json
```

Every SSMax phase installs `SSMaxHealthLedgerCallback`. OLMo Core passes every reduced train-step
metric to it even though metric collection is batched at cadence 5, and flushes those callbacks
before serializing trainer state. Each rank's checkpoint therefore contains a self-hashed,
resume-safe chain with exactly steps 1..N, finite-loss/gradient flags, optimizer-guard decisions,
and the loader's cumulative data errors. There is no W&B export or hand-authored counter input.

The health producer loads and recomputes that ledger directly from each checkpoint-bound
`train/rank*.pt`, replays the exact unpacked loader independently for every saved rank, requires
exact cursor equality, and recomputes both delivered raw and active loss mass:

```bash
PYTHONPATH=src python src/scripts/eval/vision_alignment_ssmax_perception_health.py \
  --manifest /immutable/evidence/pair-manifest.json \
  --expected-manifest-sha256 SHA256 --arm treatment --step 4000 \
  --work-dir /weka/.../health-work \
  --output /immutable/evidence/treatment-step4000-health.json
```

Promotion consumes all 12 raw receipts (evaluation and health for two arms at three steps). Pass
all `--evaluation ARM:STEP=PATH`, `--expected-evaluation-sha256 ARM:STEP=SHA256`, and corresponding
health options:

```bash
PYTHONPATH=src python src/scripts/eval/vision_alignment_ssmax_perception_promotion.py promote \
  --manifest /immutable/evidence/pair-manifest.json \
  --expected-manifest-sha256 SHA256 ... \
  --output /immutable/evidence/perception-promotion-report.json
```

The report is passed only when step-0 model states are identical; all LM tensors, non-image
embedding rows, control vision tensors, and native-text outputs stay exact; all-rank cursors and
loss mass pass; no data error, optimizer guard skip, or non-finite event occurs; the paired
source-balanced 10,000-sample bootstrap has positive lower bounds for both DID and the treatment's
absolute gap; correct-image CE is within 2% of control for the macro and every source; and the
step-4000 treatment gap retains at least 80% of step 3000. These checks apply at first 1/8/32/all.

`audit` reopens the report and exactly rebuilds it from its raw receipts. Only after that succeeds
may a human explicitly run `approve --approved-by ID --approved-at TIMESTAMP`. The resulting v5
gate permits no waivers and is accepted only for the matching SSMax treatment step 4000 when
starting joint training. There are intentionally no pre-created production profiles, manifests,
reports, approvals, or gates in this repository.

After both variants have complete rebuilt v5 reports, produce the separate descriptive model
comparison. It requires identical provenance, projection/calibration, pairings, evaluation,
topology, cadence, attention probe, and retained steps. At every treatment/control step it
directly compares the two attention reports and keeps absolute checkpoint differences separate
from the step-0-normalized adaptation quantity
`(left_step-left_step0)-(right_step-right_step0)`. It additionally reports the difference between
the two variants' treatment-vs-control adaptation DIDs. Direction labels state whether positive
means a larger visual gap or a worse correct-image CE. The schema fixes `winner: null` and is not
a promotion input:

```bash
PYTHONPATH=src python src/scripts/eval/vision_alignment_ssmax_perception_compare.py \
  --left-promotion-report /immutable/QK/perception-promotion-report.json \
  --expected-left-promotion-report-sha256 RAW_SHA256 \
  --right-promotion-report /immutable/NO_QK/perception-promotion-report.json \
  --expected-right-promotion-report-sha256 RAW_SHA256 \
  --created-at 2026-08-20T00:00:00Z \
  --output /immutable/perception-model-comparison.json
```

The comparison validator reopens both promotion-report references, rebuilds both v5 decisions from
their raw receipts, and exactly regenerates every nested bootstrap and attention comparison. A
rehashed edit to any inner descriptive result is rejected.
