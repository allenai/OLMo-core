# Direct SSMax perception evidence

This additive protocol evaluates one completed vision-unfrozen perception lineage without a
frozen-vision control. It does not reinterpret paired perception v1/v2 evidence and makes no
causal claim about the effect of unfreezing the vision encoder. The checked-in amendment fixes the
two authorized treatment runs, training revision, excluded controls, health policy, and descriptive
comparison scope.

The training Git identity remains the exact revision used by the completed checkpoints. A separate
evidence Git identity binds the later protocol, producers, and v7 consumers. Validation proves that
the evidence revision descends from training and changes only the explicit evidence/consumer
allowlist; model, data, optimizer, calibration, and training-profile sources cannot change.

## Fixed evidence population

Each lineage is evaluated at permanent steps 0, 3,000, and 4,000. The eight perception validation
sources use 480 deterministic single-response rows each. These rows are selected from the full
512-row projected validation populations and use new projection-specific wrong-image pairings;
legacy multi-response pairing files are inadmissible. On the fixed 16-rank topology, two examples
per rank produce exactly 15 complete global batches per source.

The evaluator reports correct-image and exact-geometry-wrong-image CE over first 1, 8, 32, and all
response tokens. It also verifies strict generic DCP loading, exact frozen LM/non-image embedding
surfaces, the immutable native-text sentinel, and the perception-bound attention probe. The CPU
health producer replays all 16 saved loader cursors and validates the checkpoint-native v3 health
ledger. Eligibility permits at most eight finite gradient-only guard skips, at least 128 clean
steps between skips, at least 128 final clean steps, only the genesis history reset, and zero data
or non-finite events.

Copy `ssmax_perception_direct_manifest_v1.json.template` once per model variant only after the
evidence revision is committed and clean. Replace every placeholder with an exact path/ref. Both
specs must use the same provenance, source audit, projection-specific pairings, probe, sentinel,
evaluation contract, topology, policy, and evidence Git ref.

Finalize the head manifest first; this atomically creates the shared projected pairings. Finalize
the no-QK manifest second; it must validate the same immutable files:

```bash
PYTHONPATH=src python src/scripts/eval/vision_alignment_ssmax_perception_direct_manifest.py \
  --spec /weka/.../perception-direct-v1/ssmax_head_qknorm/manifest-spec.json \
  --output /weka/.../perception-direct-v1/ssmax_head_qknorm/manifest.json
```

Launch one 2x8 Holmes evaluation for each lineage and step. The direct stage accepts no `--arm` or
checkpoint override:

```bash
PYTHONPATH=src python src/scripts/beaker_launch_vision_ssmax_evidence.py launch \
  perception_direct ssmax-head-direct-step4000 -- \
  --manifest /weka/.../perception-direct-v1/ssmax_head_qknorm/manifest.json \
  --expected-manifest-sha256 SHA256 \
  --step 4000 \
  --work-dir /weka/.../perception-direct-v1/ssmax_head_qknorm/work/step4000 \
  --output /weka/.../perception-direct-v1/ssmax_head_qknorm/step4000-evaluation.json
```

Produce the three CPU health receipts per lineage from the same manifest:

```bash
PYTHONPATH=src python src/scripts/eval/vision_alignment_ssmax_perception_direct_health.py \
  --manifest /weka/.../manifest.json --expected-manifest-sha256 SHA256 \
  --step 4000 --work-dir /weka/.../health-work/step4000 \
  --output /weka/.../step4000-health.json
```

Promotion consumes exactly six raw receipts for one lineage. A report passes only when every
receipt and health constraint passes, every frozen surface and native-text result stays exact, the
candidate correct-image CE is no more than 1.02 times its own step-0 value, the step-4,000 visual
gap retains at least 80% of step 3,000, and both the absolute step-4,000 gap and step-0-normalized
improvement have positive bootstrap lower bounds for every source and macro aggregate:

```bash
PYTHONPATH=src python src/scripts/eval/vision_alignment_ssmax_perception_direct_promotion.py \
  promote --manifest /weka/.../manifest.json --expected-manifest-sha256 SHA256 \
  --evaluation 0=/weka/.../step0-evaluation.json \
  --evaluation 3000=/weka/.../step3000-evaluation.json \
  --evaluation 4000=/weka/.../step4000-evaluation.json \
  --expected-evaluation-sha256 0=SHA256 \
  --expected-evaluation-sha256 3000=SHA256 \
  --expected-evaluation-sha256 4000=SHA256 \
  --health 0=/weka/.../step0-health.json \
  --health 3000=/weka/.../step3000-health.json \
  --health 4000=/weka/.../step4000-health.json \
  --expected-health-sha256 0=SHA256 \
  --expected-health-sha256 3000=SHA256 \
  --expected-health-sha256 4000=SHA256 \
  --output /weka/.../promotion.json
```

`audit` reopens and exactly rebuilds the report from all six receipts. Only a passed, freshly
audited report may be approved with a durable human identity and a timestamp after report creation.
The resulting waiver-free v7 gate names one exact perception step-4,000 parent.

After both reports exist, the direct comparator requires identical shared protocol inputs and emits
same-step head-minus-noQK differences plus step-0-normalized adaptation differences. It fixes
`winner: null`, is descriptive only, and is never a promotion input.

The concrete joint profiles necessarily follow the evidence approval because they bind the final
v7 gate paths and hashes. A v7 validator therefore accepts either the exact evidence checkout or
one clean descendant whose complete diff is exactly the two predeclared joint profiles and their
dedicated allowlist. No evidence producer, training input, or gate-consumer source may change in
that descendant. The evidence revision must be one non-merge commit directly after the training
revision, and the joint-profile revision must be one non-merge commit directly after the evidence
revision. Because Gantry starts from a depth-one checkout, the direct evidence launcher fetches
depth two and the v7 joint recipe fetches depth three before torchrun; validators then require the
exact three-commit chain rather than trusting an unverified descendant claim.
