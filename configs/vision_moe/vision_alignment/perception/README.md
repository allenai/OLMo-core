# Perception phase profiles

The eight canonical adapters and fail-closed artifact pipeline are implemented, but no production
profile is approved yet. Launch remains blocked until the FineVision materialization, union-
disjoint provenance, four-epoch source audit, and bridge promotion bundle/gate have been produced
and independently reviewed.

A production profile must live directly in this directory, own the complete configuration, and
have its exact raw SHA-256 added to `approved_profiles.json` in a separate review change. The
launcher persists and revalidates both the profile and allowlist identities while rejecting all
CLI overrides. These review identities are intentionally excluded from the causal data contract,
so control and treatment can prove identical data. The allowlist is intentionally empty while
artifacts are being built.

The first causal comparison uses identical parent/data/configuration across two fresh profiles:
the frozen-vision control and the vision-unfrozen treatment. Do not substitute instruction/SFT
sources or change the mixture between arms.
