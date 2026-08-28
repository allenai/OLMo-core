# Exploratory SSMax joint admission

This additive version-8 protocol may authorize matched joint alignment from a rejected strict
version-7 direct-perception report. It does not modify or waive version 7: the strict report stays
rejected, and the resulting gate is valid only for the exploratory SSMax joint phase. It cannot
select a winner, make a promotion decision, or authorize another phase or model family.

The checked-in authorization permits only source-level visual-gap deviations in `first_1` or
`first_8` response windows. Every macro aggregate, every `first_32` and all-token source result,
every correct-image CE result, and every technical or health constraint must still pass. A report
with no deviations uses version 7 instead. The validator reopens the pinned rejected report,
rebuilds it exactly from all six raw receipts, revalidates the live step-4,000 checkpoint and Git
lineage, and records every accepted deviation verbatim in the version-8 gate.

Audit one immutable rejected report before approval:

```bash
PYTHONPATH=src python \
  src/scripts/eval/vision_alignment_ssmax_perception_exploratory.py audit \
  --report /weka/.../perception-direct-v4/ARM/promotion.json \
  --expected-report-sha256 REPORT_SHA256
```

After both the strict report and checked-in authorization exist, issue the research-only gate with
the durable authorized identity and a timestamp no earlier than either artifact:

```bash
PYTHONPATH=src python \
  src/scripts/eval/vision_alignment_ssmax_perception_exploratory.py approve \
  --report /weka/.../perception-direct-v4/ARM/promotion.json \
  --expected-report-sha256 REPORT_SHA256 \
  --approved-by rustins \
  --approved-at YYYY-MM-DDTHH:MM:SSZ \
  --output /weka/.../perception-direct-v4/ARM/exploratory-parent-gate-v8.json
```

Create both concrete joint profiles together in one clean commit directly after the evidence
revision. Each profile must bind its own step-4,000 perception checkpoint and exact version-8 gate
while retaining the same joint recipe, native replay, visual data, seeds, topology, and 16,000-step
schedule. The joint checkout must be the sole exact child of the evidence revision and may change
only those two profiles and their dedicated allowlist. Gantry fetches depth four so validation can
prove the complete training -> base evidence -> exploratory evidence -> joint-profile chain.

Joint evaluation remains descriptive. Retain steps 0, 4,000, 8,000, 12,000, and 16,000; compare
BLINK Jigsaw, MathVista geometry, correct-image CE and matched-wrong visual gap, native-text
retention, attention-logit magnitude, entropy/effective context, and image/prompt/response routing
trajectories. Neither a directional result nor a null result is a promotion decision.
