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
provenance preflight. Submit only through the project launch path fixed to workspace
`ai2/molmofication`; never use a default or alternate Beaker workspace.
