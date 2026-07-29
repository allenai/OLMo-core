# LatentMoE qualification

## Implementation source and audit

The implementation is the LatentMoE series from
[OLMo-core PR #799](https://github.com/allenai/OLMo-core/pull/799), ported onto
`jacobm/moe-v2-core-gdn2` without merging the PR branch's unrelated upstream
history. It covers legacy MoE, MoEHybrid, and OLMo-DDP MoE-v2 execution paths,
including no-EP, 1D EP, rowwise EP, rowwise-wave, and DeepEP. The port retains
this branch's existing EP1 MXFP8 no-EP behavior at the one overlapping call
site. The PR's latest router fix changes the router input from the projected
expert input to the full-width token representation, matching the paper and
Kimi K3. We synchronized that implementation at PR head
`967abef8cf835e8076672cc622b9d9e866f26908`. The material source-level API
change is the rename from `routed_expert_dim` to `latent_dim`; tensor parameter
names, model shapes, and the previously audited parameter counts are
unchanged. Older serialized configs using the former field name must be
updated, while their checkpoint tensors remain compatible.

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

The earlier PR implementation routed on the down-projected representation,
whereas the LatentMoE paper keeps the routing gate at full model width. The
latest PR fix and this branch now both use the paper behavior. The paper's
accuracy-oriented recipe also increases total experts and top-k by the
compression factor. We retain fixed-expert controls to isolate latent
projection effects and add a paper-matched arm that scales both quantities.

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
| LatentMoE 4× EP1 approximation | 640 → 160 | 1,000 / 32 | 296,632,128 | 232,406,848 | 3,073,320,768 | +2.110% |

The 4× compression matches the setting selected in the
[LatentMoE paper](https://arxiv.org/abs/2601.18089); the 2× point is the
conservative control. Multiplying total experts preserves stored routed-expert
parameters, while multiplying top-k preserves active routed-expert parameters.
The exact paper-scaled 4× point has 1,024 routed experts and therefore requires
EP2 with the current grouped expert kernel. The 1,000-expert approximation
changes no other architectural setting, keeps top-32 routing, runs on EP1, and
is the 4× production candidate.

Model variants:

- `geometry_275m_kda_ev2_neg_nope_gated_latent2x_fullrouter`
- `geometry_275m_kda_ev2_neg_nope_gated_latent4x_fullrouter`
- `geometry_275m_kda_ev2_neg_nope_gated_latent2x_papermatched`
- `geometry_275m_kda_ev2_neg_nope_gated_latent4x_papermatched`
- `geometry_275m_kda_ev2_neg_nope_gated_latent4x_1000experts`

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

The final throughput protocol uses 50 optimizer steps and the median of steps
41--50. The matched non-latent `L=1` control is EP1/MB16 on one B300:
[L=1 control](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYP31SFS8DYN7GJHX2CTDHT1).
The initial LatentMoE MB16 attempt OOMed during the compiled dry-run and is
superseded by an explicit capacity search. At a one-microbatch batch, 2×
physically fits MB15 but not MB16; 4× fits MB9 but not MB10. However, 4× MB9
OOMs on the second accumulated microbatch because persistent gradient and
accumulation buffers are then live. Its usable large-batch maximum is therefore
MB8. Future capacity tests must execute at least two accumulated microbatches;
a one-microbatch dry-run is insufficient.

Final exact-2-Mi measurements for the original qualification candidates:

| Variant | Topology | Rank MB / accumulation | TPS/GPU | Aggregate TPS | TFLOPs/GPU | Step seconds | Active / reserved GiB |
|---|---|---:|---:|---:|---:|---:|---:|
| KDA parent (`L=1`) | 1 B300, EP1 | 16 / 16 | 255,489.0 | 255,489.0 | 388.60 | 8.2084 | 235.70 / 239.20 |
| LatentMoE 2× paper-matched | 1 B300, EP1 | 8 / 32 | 214,671.5 | 214,671.5 | 333.15 | 9.7691 | 162.90 / 164.50 |
| Mixed 3-KDA/2-SWA/1-FA + LatentMoE 2× | 1 B300, EP1 | 8 / 32 | 215,204.5 | 215,204.5 | 351.55 | 9.7449 | 167.50 / 169.20 |
| LatentMoE 4× paper-matched | 2 B300s, EP2 | 8 / 16 | 143,012.5 | 286,025.0 | 222.90 | 7.3321 | 197.50 / 214.20 |
| LatentMoE 4×, 1,000 experts | 1 B300, EP1 | 8 / 32 | 179,997.0 | 179,997.0 | 280.40 | 11.6510 | 199.00 / 202.00 |

The 2× physical-max point uses MB15/accumulation-17 and the nearest lower
batch to 2 Mi, 2,088,960 tokens. It reaches 219,975.0 TPS/GPU and 341.40
TFLOPs/GPU: 2.47% faster than the exact-2-Mi MB8 point, but still 13.90% below
the `L=1` TPS. The exact-2-Mi 2× point is 15.98% below `L=1`. The 4× point is
44.02% lower per GPU than `L=1`; its 286,025 aggregate TPS is 11.95% higher
only because it uses two GPUs, so it is not a resource-efficiency win.
At the exact-2-Mi MB8 setting, replacing the ordinary KDA mixer pattern with
the deeper mixed-attention motif changes raw TPS by only `+0.25%`, while peak
active memory rises 4.6 GiB.

All accepted rows have 50 metric-bearing steps and zero skipped optimizer
steps. The 4× job logged `Training complete` after step 50, then exited nonzero
from a DataLoader/torchrun shutdown segfault; its measurements are complete and
the failure is not a training or capacity failure.

For the selected 1,000-expert EP1 replacement, MB11 is the exact one-B300
ceiling: MB9, MB10, and MB11 each completed the compiled dry run and all six
accumulated optimizer steps, while MB12--16 OOMed. MB11 peaked at 247.1 GiB
active and 256.3 GiB reserved in the six-step bracket on the 267.7-GiB B300.
Production deliberately uses MB8/accumulation-1 at Cx1 and
MB6/accumulation-2 at Cx2 because those are the largest legal factorizations
of their standard global batches.

The formal physical-ceiling measurement uses MB11/accumulation-23 and the
nearest lower batch to 2 Mi, 2,072,576 tokens. It reaches 182,898.5 TPS/GPU and
284.90 TFLOPs/GPU, 1.61% more TPS than the exact-2-Mi MB8 point, with 254.2 GiB
active / 257.9 GiB reserved. Both 50-step runs have zero skipped optimizer
steps.

- [Capacity sweep](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYP4ZJGFTKBVHBEKW6GCZV58)
- [Exact-2-Mi final measurements](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYP5SXH7ZK32FA8NM7AHXRV0)
- [Physical-max near-2-Mi measurements](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYP6RYG25M5VDSMDQ2S0ZRTR)
- [Machine-readable results](results/throughput/275m_kda_latent_moe_paper.csv)
- [1,000-expert capacity bracket](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYP96YAGZ4CZCZBT87NHSA5X)
- [1,000-expert MB9 completion](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYPARNB9QDWBGV5HE3PKDK76)
- [1,000-expert exact-2-Mi throughput](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYPA1GYA09H78GCVT3PQ1TZH)
- [1,000-expert MB11 throughput](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYPBK8K72R636K3H4SJ13G5X)

## Initial full LR sweep

The first production wave covers Cx1 and Cx2 for the 2× paper-matched family
and the 4×/1,000-expert EP1 approximation at `4e-4`, `8e-4`, `1.6e-3`, and
`3.2e-3` (16 jobs total). Every job uses four Holmes B300s, urgent priority, a
two-hour allocated `minRuntime`, automatic checkpoint resume, 500-step
ephemeral saves, no in-loop evaluation, and no permanent intermediate
checkpoints. The optimizer batches remain the standard 262,144 tokens at Cx1
and 393,216 tokens at Cx2.

| Family | Cx | Topology | Rank MB / accumulation |
|---|---:|---|---:|
| L=2, 512/top-16 | 1 | 4 GPUs, EP1 | 8 / 1 |
| L=2, 512/top-16 | 2 | 4 GPUs, EP1 | 12 / 1 |
| L=4, 1000/top-32 | 1 | 4 GPUs, EP1 | 8 / 1 |
| L=4, 1000/top-32 | 2 | 4 GPUs, EP1 | 6 / 2 |

The 2× source manifest is
[`launchers/pretraining/manifests/275m_kda_latent_moe_lr_sweep_cx1_cx2.yaml`](launchers/pretraining/manifests/275m_kda_latent_moe_lr_sweep_cx1_cx2.yaml).
The original 1,024-expert 4× tasks in that submission were canceled before
training. Their EP1 replacements are generated from
[`launchers/pretraining/manifests/275m_kda_latent_moe_l4_1000e_lr_sweep_cx1_cx2.yaml`](launchers/pretraining/manifests/275m_kda_latent_moe_l4_1000e_lr_sweep_cx1_cx2.yaml).
The replacement sweep is
[Beaker experiment `01KYPC428EW5CZFE2X8VZJ2QMQ`](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYPC428EW5CZFE2X8VZJ2QMQ).
The paired `plot_kda_latent_moe.py` entry point publishes separate `L=2` and
`L=4` U-plots plus one strict best-of comparison against the BF16 KDA parent.
Cx4/Cx8 exact names are reserved in the registry but remain unlaunched.

Final status on 2026-07-29: all 16 launched Cx1/Cx2 cells finished and both
families have complete, bracketed curves. Three original L2 jobs had terminal
post-training bookkeeping failures; their exact replacements completed in
[01KYP9CDR7QV5Y9Z88M0CZQ0TC](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYP9CDR7QV5Y9Z88M0CZQ0TC).
The observed winner is `1.6e-3` for both Cx values and both families:

| Family | Cx1 final-250M CE | Cx2 final-250M CE |
|---|---:|---:|
| L=2, 512/top-16 | `2.659602` | `2.541918` |
| L=4, 1,000/top-32 | `2.663658` | `2.555501` |

The refreshed U-plots, best-of plot, and machine-readable results live under
[`plots/pretraining/kda_latent_moe`](plots/pretraining/kda_latent_moe) and
[`results/pretraining/kda_latent_moe`](results/pretraining/kda_latent_moe).

## Validation hold

Do not launch new validation backfills while the current capacity constraint is
in effect. The earlier 48-ready/five-training audit is stale; re-audit and
de-duplicate against the finished 117-target historical registry before
submission. The intended policy remains final-checkpoint LR winners only, EP1,
LM validation plus the `fast` downstream suite. The completed LatentMoE sweep
now contributes four selected targets: L2 and L4 at Cx1 and Cx2, all at
observed LR `1.6e-3`.
