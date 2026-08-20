# Perception phase profiles

The eight canonical adapters, FineVision materialization, union-disjoint provenance, four-epoch
source audit, and bridge promotion bundle/gate have been produced and independently reviewed. The
frozen-vision control and vision-unfrozen treatment profiles are approved by exact raw SHA-256 in
`approved_profiles.json`.

A production profile must live directly in this directory, own the complete configuration, and
have its exact raw SHA-256 added to `approved_profiles.json` in a separate review change. The
launcher persists and revalidates both the profile and allowlist identities while rejecting all
CLI overrides. These review identities are intentionally excluded from the causal data contract,
so control and treatment can prove identical data.

The first causal comparison uses identical parent/data/configuration across two fresh profiles:
the frozen-vision control and the vision-unfrozen treatment. Do not substitute instruction/SFT
sources or change the mixture between arms.

Before either training launch, run the pinned profile-pair auditor and the exact two-node runtime
provenance preflight. Historical s002 profiles remain fixed to `ai2/molmofication`; SSMax profiles
are fixed to `ai2/scaling-ladders`. Never use a default or alternate Beaker workspace.

## SSMax 1.4B Cx8 pairs

The QK-norm and no-QK-norm lineages each require their own two-profile causal fork from that
lineage's approved permanent bridge checkpoint: one `treatment` and one
`frozen_vision_control`. Within a lineage, the parent checkpoint/config/gate, visual provenance,
source audit, loss-mass calibration, duration, seeds, checkpoint cadence, and evaluation cadence
must be identical. The control transformation is derived by the recipe and consists only of
adding `vision.*` to the freeze patterns and setting the vision optimizer-group LR to zero. The
runtime trainability check independently proves that every vision parameter is frozen while the
connector and six input image-token rows retain the treatment configuration.

The SSMax loss-mass calibration is distinct from the historical multi-response source audit.
Build it with `src/scripts/data/build_ssmax_single_response_calibration.py` after provenance
selection and pin its raw SHA plus every projected mean in both profiles. It uses data seed 95818
to choose a branch from the stable underlying raw index; evaluation is fixed to backing epoch
zero. Pairing/bootstrap seed 6198 is a separate evidence choice. A profile without this immutable
receipt, its complete 512-row/source validation preflight, or exact projected means is
non-runnable.

Do not add placeholder SSMax profiles to `approved_profiles.json`. Create the four concrete YAML
profiles only after the bridge checkpoints and human-approved SSMax promotion gates exist, then
review and allowlist their exact bytes. They must select `ssmax_head_qknorm` or
`ssmax_no_qknorm`, use workspace `ai2/scaling-ladders`, and may reuse the existing visual
provenance and source-audit artifacts because those artifacts describe data rather than a model
lineage.

For each concrete SSMax pair, write the immutable auditor output as
`artifacts/ssmax-head-qknorm-perception-profile-pair-v3.json` or
`artifacts/ssmax-no-qknorm-perception-profile-pair-v3.json`. Run the distributed preflight with
the matching explicit `--model-variant=ssmax_head_qknorm` or
`--model-variant=ssmax_no_qknorm`. The v3 path fails closed unless both profiles select the same
lineage and differ only by the derived vision freeze/LR intervention; it also requires dense
generic BF16/FP32 HSDP, no expert parallelism or router loss, and permanent checkpoints at steps
500, 1000, 2000, 3000, and 4000. The historical s002 v2 schema and lineage policy remain intact.

The executable post-hoc causal-pair protocol and non-runnable spec templates are documented in
[`../eval/SSMAX_PERCEPTION_EVALUATION.md`](../eval/SSMAX_PERCEPTION_EVALUATION.md). It has its own
manifest, receipts, report, and v5 approval schema and does not inherit s002 skip lists or waivers.
