# Historical perception profiles

[frozen_vision_control_v1.yaml](frozen_vision_control_v1.yaml) and
[treatment_v1.yaml](treatment_v1.yaml) retain the reviewed legacy `Vision-Alignment.py`
causal comparison. Their exact raw SHA-256 identities are in
[approved_profiles.json](approved_profiles.json). Use the
[API guide](../../../../docs/source/guides/vision_alignment.md) for native training;
these source/profile hash gates and approval receipts are historical compatibility rules.

Both profiles pin the bridge step500 parent and approved v2 gate, eight-source
union-disjoint provenance, FineVision materialization and four-epoch source audit.
The control freezes vision; the treatment unfreezes it. Parent, data, mixture and all
non-treatment settings must match; instruction/SFT sources are not substitutes.
Exact inputs and evidence paths are in the YAML under
`/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/`.

The legacy launcher requires complete profiles directly in this directory, separately
reviewed raw hashes in the allowlist and no CLI overrides. It persists and revalidates
both identities; those review identities are excluded from the causal data contract.
Do not edit pinned profiles or archived runtime inputs to reuse an existing approval.

Historical launch validation includes `src/scripts/eval/vision_alignment_perception_profile_pair.py`
and a two-node runtime provenance preflight. The latter script is no longer in this checkout;
reproducing that procedure requires the original deployment, not a current native launch.
Beaker submissions use the project wrapper and `ai2/molmofication` only. See the
[experiment index](../README.md) for the archived procedure and evidence pointers.
