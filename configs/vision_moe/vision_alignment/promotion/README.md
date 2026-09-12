# Historical bridge promotion evidence

This directory retains the receipt contract for `vision-alignment-bridge-real-v1/step500`.
It supports legacy `Vision-Alignment.py` consumers, not the supported native recipe described
in the [API guide](../../../../docs/source/guides/vision_alignment.md). Source-hash checks,
profile approvals and parent gates below are historical compatibility requirements.

The saved bundle and approved v2 parent gate are under
`/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/evals/bridge-real-v1-promotion-v1/`.
Their presence is not a fresh audit. See the [experiment index](../README.md) for the archive
of exact commands, artifact hashes and original evidence notes.

## Required evidence

Use the matching frozen source revision and pinned parent/data artifacts. The archived recipe
hash is not the current checkout's hash; do not replace evidence pins to bypass a mismatch.
All outputs are immutable: inspect and pin existing files instead of overwriting them.

| Receipt | Required inputs and checks |
| --- | --- |
| Text sentinel | Bare s002 parent config and exact expanded pretraining manifest; 128 deterministic 256-token windows. Requires credentials for S3 range reads. |
| Frozen state and text retention | Bridge step0/step500 DCPs, pinned matched-step500 receipt and sentinel; same native EP8 backend. Check complete loads, all 806 frozen tensors and non-image embedding rows; finite token CE, absolute/relative deltas at most `1e-6`, identical argmax tokens. |
| Cumulative loss mass | Matching recipe, audited datasets and all 16 trainer-rank states; replay all 500 batches/rank. Require exact version-5 cursor equality, zero data errors and raw/active caption/transcript shares within two percentage points of 70/30. |
| Independent matched pairs | Pinned primary pairings, exclusions, seeds and step0/step500 checkpoints. Exclude both primary recipients and donors; reuse the new pairings across checkpoints. |
| Optimizer/run health | All trainer-rank states, permanent checkpoint markers and pinned raw run log. The recorded step356 guard skip requires the explicit waiver below. |

Independent step0 must reproduce the null for every `first_8`, `first_32` and `all`
confidence interval. Step500 requires positive lower bounds for all six source/window
intervals, at least 80% of each primary `first_8`/`first_32` gap, and correct-image CE
within +2% of primary step500. Exact loader replay is an expensive CPU/image-preprocessing
job; state/text and matched-pair measurements require the native EP8 GPU environment.

The receipt producers are `src/scripts/eval/vision_alignment_state_text.py`,
`vision_alignment_loss_mass.py`, `vision_alignment_matched_wrong.py`, and the
`text-sentinel`/`run-health` commands in `vision_alignment_promotion.py` in that directory.
The promotion tool's `build` command consumes these receipts plus the canary step250,
bridge step250/step500 and independent step0/step500 matched receipts; it does not create
missing measurements. Consult the archived commands for their exact input/output paths.

## Audit and approval

From the repository root, with the independently recorded raw bundle SHA-256:

```bash
alignment_root=/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment
PYTHONPATH=src python src/scripts/eval/vision_alignment_promotion.py audit \
  --bundle="$alignment_root/evals/bridge-real-v1-promotion-v1/promotion-bundle.json" \
  --expected-sha256="${PROMOTION_BUNDLE_SHA256:?Set the independently recorded bundle hash}" \
  --expected-checkpoint="$alignment_root/checkpoints/vision-alignment-bridge-real-v1/step500" \
  --expected-checkpoint-config-sha256=41df40c299f4f3101c3ef58d657d99fb624194beaee7321ea456727212be1dad
```

Legacy production perception requires passing receipts, a successful bundle audit and an
accountable human's v2 approval naming exactly `step250_caption_first32_90pct_canary` and
`step356_optimizer_guard_skip`. The `approve` command requires the pinned bundle hash,
human identity, UTC timestamp and both explicit waiver IDs. Pin the resulting gate path
and raw hash in `initialization.parent_gate_path` and `initialization.parent_gate_sha256`.
V1 gates remain limited to old/non-production test flows; none of this is a native launch gate.

Any Beaker replay must use an experiment spec submitted through
`python src/scripts/beaker_submit_vision_moe.py SPEC.yaml --name NAME` to
`ai2/molmofication`, never direct `beaker experiment create` or another workspace.
