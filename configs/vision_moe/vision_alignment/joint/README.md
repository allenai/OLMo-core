# Historical joint profile

[joint_v1.yaml](joint_v1.yaml) is the reviewed legacy `Vision-Alignment.py` joint recipe,
retained with its [profile allowlist](approved_profiles.json) for historical consumers.
Use the [API guide](../../../../docs/source/guides/vision_alignment.md) for native training;
legacy profile hashes and approval receipts are not requirements of that pipeline.

The profile pins the perception treatment step4000 parent and its approval gate, the
8,192-token visual projection/source audit, and the s002 compact native replay train/holdout
pair. Exact paths and hashes are in the YAML. Artifacts are under
`/weka/oe-training-default/rustin/experiments/vision-moe/vision-alignment/`, including
`artifacts/s002-compact-replay-v3/` and `evals/perception-v1-promotion-v1/`.

The replay receipt binds the parent's exact 950 ordered paths and dataset fingerprint to
its config, mixture, trainer state, remote object generations and consumed byte ranges.
The LM is unfrozen in joint, so native replay and a disjoint replay holdout are mandatory
for this historical profile; Tulu/chat-formatted text is not a replacement.

Preserve the profile, allowlist and matching source deployment when reopening old runs.
Their reviewed identities do not establish compatibility with changed source or data.
See the [experiment index](../README.md) for evidence locations and the original README archive.
