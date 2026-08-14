# Joint phase profiles

`joint_v1.yaml` is the reviewed initial production recipe. It pins the approved permanent
perception checkpoint, the 8,192-token visual projection and audit, and the compact s002 native
replay train/holdout pair.

The compact replay artifact is derived self-service from the checkpoint's exact 950 ordered s002
paths and saved dataset fingerprint. Its immutable v3 receipt binds the config, paths, mix,
trainer state, exact remote object generations, and every consumed byte range; it does not depend
on an unverifiable owner-authored inventory.

The language model is unfrozen only in this phase, so native replay and its disjoint holdout are
mandatory; Tulu or chat-formatted text is not an allowed replacement.
