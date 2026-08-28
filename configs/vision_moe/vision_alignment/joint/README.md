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

## Shared replay for the SSMax pair

Both SSMax parents save the same ordered 950 OLMo-mix-0925 objects as s002. Their path manifests
use `gs://ai2-llm/...` while s002 uses the storage-equivalent `s3://ai2-llm/...`; after normalizing
only that scheme alias, every bucket/key row is identical. The parent configs have the same native
FSL dataset semantics, and each restricted rank-0 trainer state records the same v2.0 dataset
fingerprint, tokenizer, instance filter, and 8,192-token sequence contract.

The recipe re-proves those pinned config, path, and trainer-state identities before allowing an
SSMax joint phase to reference the existing immutable s002 compact-v3 train/holdout pair. This
shares the exact replay windows across QK arms without conflating their model checkpoints or
reusing optimizer/data cursors. Any corpus, ordering, fingerprint, or dataset-config difference
fails closed.

The generic dense-HSDP model-only loader and shared replay path are ready. Strict SSMax joint
admission requires each direct perception treatment to pass the no-control protocol and receive a
waiver-free, lineage-bound v7 gate. A separately authorized rejected report may instead receive a
research-only v8 gate when its deviations are limited to the checked-in short-prefix source policy.
That path preserves the v7 rejection and cannot make a promotion or winner decision. Historical
paired perception v5/v6 gates remain accepted through their original versioned validators, but the
current direct program does not require or infer a frozen-vision control. The existing v3 perception
gate authenticates one exact s002 experiment and cannot authorize either SSMax lineage.

Only after the direct perception evidence is produced and explicitly human-approved should one
concrete joint profile per SSMax treatment lineage be added to `approved_profiles.json`. Each
profile must retain its own exact model parent and version-matched v7 or v8 gate while pointing both
lineages at the same reviewed compact replay paths and pins. Add both profiles atomically; do not
add placeholder profiles or gates.

The generic post-hoc joint evidence implementation and deliberately non-runnable per-lineage
manifest templates are documented in
[`../eval/SSMAX_JOINT_EVIDENCE.md`](../eval/SSMAX_JOINT_EVIDENCE.md). They bind steps
0/4000/8000/12000/16000 and remain descriptive-only; they do not authorize joint training before
the required live lineage-matched v7 or v8 perception approvals, and they do not choose or launch
mid-training.
