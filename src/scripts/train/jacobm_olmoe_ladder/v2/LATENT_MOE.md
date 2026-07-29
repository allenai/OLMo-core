# LatentMoE qualification

## Implementation source and audit

The implementation is the nine-commit LatentMoE series from
[OLMo-core PR #799](https://github.com/allenai/OLMo-core/pull/799), ported onto
`jacobm/moe-v2-core-gdn2` without merging the PR branch's unrelated upstream
history. It covers legacy MoE, MoEHybrid, and OLMo-DDP MoE-v2 execution paths,
including no-EP, 1D EP, rowwise EP, rowwise-wave, and DeepEP. The port retains
this branch's existing EP1 MXFP8 no-EP behavior at the one overlapping call
site.

The implementation is structurally coherent:

- it validates that the latent dimension is positive, smaller than the model
  dimension, and equal to both the routed-expert and routed-router input size;
- it down-projects only the routed branch, restores it to model width before
  combining it with the full-width shared expert, and leaves the residual
  stream unchanged;
- it includes the latent projections in initialization, parameter/FLOP counts,
  TP plans, serialization, and every supported MoE-v2 dispatch path; and
- it explicitly rejects the unsupported LatentMoE + rowwise-TBO combination.

One scientific distinction must remain visible. PR #799 routes on the
down-projected representation. The LatentMoE paper's default formulation keeps
the routing gate at the full model width, and its accuracy-oriented recipe also
increases total experts and top-k by the compression factor. The initial ladder
qualification intentionally tests the PR behavior as written and holds expert
count/top-k fixed. Before a full ladder, decide whether to retain that isolated
variant or add a separate paper-exact router/expert-scaling recipe.

The current 320/160 latent widths are qualified for EP1/no-EP first. PR #799's
DeepEP BF16 path requires the routed hidden dimension to be divisible by 256,
so these exact small-model widths must not be carried into DeepEP runs. If a
larger LatentMoE rung needs DeepEP, select a 256-aligned latent width or use a
different supported EP path and smoke-test that layout independently.

## First 275M candidates

Both candidates start from the promoted 275M KDA recipe: 640 model width, 10
layers, KDA `expand_v=2` with negative eigenvalues, gated NoPE full-attention
layers, 664 expert hidden width, 64 routed experts, top-8, and one full-width
shared expert. Only the routed latent width changes. The optional RMSNorm before
the latent up projection is explicitly disabled for these first candidates,
matching the PR default and avoiding a second intervention.

| Candidate | Model → latent | Compression | Active params | Active non-embedding | Stored params | Active change vs KDA parent |
|---|---:|---:|---:|---:|---:|---:|
| KDA parent | 640 → 640 | 1× | 290,503,488 | 226,278,208 | 3,136,035,648 | — |
| LatentMoE 2× | 640 → 320 | 2× | 247,556,928 | 183,331,648 | 1,670,323,008 | -14.783% |
| LatentMoE 4× | 640 → 160 | 4× | 222,397,248 | 158,171,968 | 933,780,288 | -23.444% |

The 4× point matches the compression selected in the
[LatentMoE paper](https://arxiv.org/abs/2601.18089); the 2× point is the
conservative control. These are not active-parameter-matched models because
expert count/top-k remain fixed.

Model variants:

- `geometry_275m_kda_ev2_neg_nope_gated_latent2x`
- `geometry_275m_kda_ev2_neg_nope_gated_latent4x`

The minimal smoke manifest is
[`launchers/pretraining/manifests/275m_kda_latent_moe_smokes.yaml`](launchers/pretraining/manifests/275m_kda_latent_moe_smokes.yaml).
It runs 12 compiled optimizer steps per candidate on one Holmes B300 with EP1,
MB4/accumulation-2, a 30-minute allocated runtime, no W&B, no evaluation, and
no checkpoint writes. The first unallocated submission was canceled before
startup because the workspace's 64 unallocated slots were already full.

## Validation hold

Do not launch new validation backfills while the current capacity constraint is
in effect. The last audited winner-only backlog had 48 checkpoints ready and 5
KDA checkpoints still training; re-audit completion and de-duplicate against
the finished 117-target historical registry before submission. The intended
policy remains final-checkpoint LR winners only, EP1, LM validation plus the
`fast` downstream suite. LatentMoE has no validation target until a full
training sweep produces a selected winner.
