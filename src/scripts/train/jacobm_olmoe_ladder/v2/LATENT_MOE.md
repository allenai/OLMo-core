# LatentMoE qualification

## Implementation source and audit

The implementation is the nine-commit LatentMoE series from
[OLMo-core PR #799](https://github.com/allenai/OLMo-core/pull/799), ported onto
`jacobm/moe-v2-core-gdn2` without merging the PR branch's unrelated upstream
history. It covers legacy MoE, MoEHybrid, and OLMo-DDP MoE-v2 execution paths,
including no-EP, 1D EP, rowwise EP, rowwise-wave, and DeepEP. The port retains
this branch's existing EP1 MXFP8 no-EP behavior at the one overlapping call
site. Our follow-up changes the router input from the projected expert input to
the full-width token representation, matching the paper and Kimi K3.

The implementation is structurally coherent:

- it validates that the latent dimension is positive, smaller than the model
  dimension, and equal to the routed-expert input size, while the routed router
  remains at model width;
- it down-projects only the routed branch, restores it to model width before
  combining it with the full-width shared expert, and leaves the residual
  stream unchanged;
- it includes the latent projections in initialization, parameter/FLOP counts,
  TP plans, serialization, and every supported MoE-v2 dispatch path; and
- it explicitly rejects the unsupported LatentMoE + rowwise-TBO combination.

One scientific distinction remains visible. PR #799 routes on the
down-projected representation, whereas the LatentMoE paper keeps the routing
gate at full model width. We qualify both semantics independently. The paper's
accuracy-oriented recipe also increases total experts and top-k by the
compression factor. We retain fixed-expert controls to isolate the router-input
correction and add a third, paper-matched arm that scales both quantities.

Kimi K3 independently confirms the full-width-router design in released code:
it computes routing from the 7,168-wide residual representation, then projects
only the routed expert input to 3,584 dimensions. It enables a pre-up-projection
RMSNorm and uses 896 routed experts with top-16. Treat enabling the norm and
scaling expert-count/top-k as separately visible choices rather than folding
them into the router-input correction.

The current 320/160 latent widths are qualified for EP1/no-EP first. PR #799's
DeepEP BF16 path requires the routed hidden dimension to be divisible by 256,
so these exact small-model widths must not be carried into DeepEP runs. If a
larger LatentMoE rung needs DeepEP, select a 256-aligned latent width or use a
different supported EP path and smoke-test that layout independently.

## First 275M candidates

All candidates start from the promoted 275M KDA recipe: 640 model width, 10
layers, KDA `expand_v=2` with negative eigenvalues, gated NoPE full-attention
layers, 664 expert hidden width, 256 routed experts, top-8, and one full-width
shared expert. The fixed-expert controls change only routed latent width and
router semantics. The paper-matched arm additionally multiplies both total
experts and top-k by compression. The optional RMSNorm before the latent up
projection remains disabled, matching the PR default and avoiding another
intervention.

| Candidate | Model → latent | Experts / top-k | Active params | Active non-embedding | Stored params | Active change vs KDA parent |
|---|---:|---:|---:|---:|---:|---:|
| KDA parent | 640 → 640 | 256 / 8 | 290,503,488 | 226,278,208 | 3,136,035,648 | — |
| LatentMoE 2× full-router control | 640 → 320 | 256 / 8 | 248,294,208 | 184,068,928 | 1,671,060,288 | -14.530% |
| LatentMoE 4× full-router control | 640 → 160 | 256 / 8 | 223,503,168 | 159,277,888 | 934,886,208 | -23.064% |
| LatentMoE 2× paper-matched | 640 → 320 | 512 / 16 | 295,664,448 | 231,439,168 | 3,141,196,608 | +1.777% |
| LatentMoE 4× paper-matched | 640 → 160 | 1,024 / 32 | 296,770,368 | 232,545,088 | 3,142,302,528 | +2.157% |

The 4× point matches the compression selected in the
[LatentMoE paper](https://arxiv.org/abs/2601.18089); the 2× point is the
conservative control. Multiplying total experts preserves stored routed-expert
parameters, while multiplying top-k preserves active routed-expert parameters.

Model variants:

- `geometry_275m_kda_ev2_neg_nope_gated_latent2x_fullrouter`
- `geometry_275m_kda_ev2_neg_nope_gated_latent4x_fullrouter`
- `geometry_275m_kda_ev2_neg_nope_gated_latent2x_papermatched`
- `geometry_275m_kda_ev2_neg_nope_gated_latent4x_papermatched`

The fixed-expert smoke manifest is
[`launchers/pretraining/manifests/275m_kda_latent_moe_smokes.yaml`](launchers/pretraining/manifests/275m_kda_latent_moe_smokes.yaml).
The paper-matched smoke manifest is
[`launchers/pretraining/manifests/275m_kda_latent_moe_paper_smokes.yaml`](launchers/pretraining/manifests/275m_kda_latent_moe_paper_smokes.yaml).
It runs 12 compiled optimizer steps per candidate on one Holmes B300 with EP1,
MB4/accumulation-2, a 30-minute allocated runtime, no W&B, no evaluation, and
no checkpoint writes. The first unallocated submission was canceled before
startup because the workspace's 64 unallocated slots were already full. The
first allocated PR-semantics attempt was mistakenly canceled after startup and
is inconclusive. Its exact two-task replacement is pinned to detached commit
`7490a4a5a`; the full-width-router pair is pinned independently so the two arms
cannot read each other's source checkout:

- [PR #799 routing semantics](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYNZRH7S8APE059CR042BTPC)
- [full-width routing semantics](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYP02F6BRDCQSDR8JWDATAK8)
- [paper-matched expert scaling](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYP0NWK5P0R9GQ22HS0P6BFH)

The paper-matched 2× EP1 task passed. The first paper-matched 4× EP1 task
reached the grouped-kernel limit because the CUDA implementation requires
`group_count < 1024`, despite its error saying it cannot process *more than*
1,024 groups. Exactly 1,024 routed experts therefore fails on EP1; the shared
expert is unrelated and runs through a separate dense path. Its qualification
replacement uses EP2, leaving 512 routed experts per rank while preserving the
exact global model and batch:
[4× EP2 replacement](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYP2ESY2RPK1PBHTK42VEGPZ).

The corresponding 50-step throughput qualification uses a 2 Mi-token batch,
MB16, and final-10 medians:
[latent throughput work](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYP2RYXXK398CRZMB94ARXAK).
The matched non-latent `L=1` control uses the same single-B300 protocol:
[L=1 control](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYP31SFS8DYN7GJHX2CTDHT1).

## Validation hold

Do not launch new validation backfills while the current capacity constraint is
in effect. The last audited winner-only backlog had 48 checkpoints ready and 5
KDA checkpoints still training; re-audit completion and de-duplicate against
the finished 117-target historical registry before submission. The intended
policy remains final-checkpoint LR winners only, EP1, LM validation plus the
`fast` downstream suite. LatentMoE has no validation target until a full
training sweep produces a selected winner.
