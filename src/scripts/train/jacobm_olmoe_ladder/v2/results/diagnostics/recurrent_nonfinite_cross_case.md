# Cross-case GDN non-finite analysis

Status: cross-case replay complete (2026-07-26)

## Established canonical-GDN2 result

The canonical GDN2 480M Cx2 step-24668 failure is a forward recurrent-state
blow-up, not an optimizer, distributed-reduction, recomputation, or Triton-only
failure. The exact trained layer and input diverge in both the production chunk
kernel and a token-by-token FP32 recurrence. The sequential reference first
becomes non-finite at token 7039, while the chunk kernel exposes the failure at
the following 64-token boundary (token 7104). The triggering 8K sequence is an
extremely repetitive `stack-edu_Cpp` sample: 12 unique token IDs and 2294/8128
lag-64 matches.

## Representative replays

| Case | Prior symptom | Exact source | Diagnostic |
|---|---|---|---|
| Original GDN2 expand_v=2, 1.2B Cx4 | Step 9059; first failure is rank 5, block-0 GDN2 forward at token 4992; broad gradients are downstream; reproduced with backward recomputation disabled | step9000 | [01KYFWDECQSBGYCH5WFP5QDE30](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYFWDECQSBGYCH5WFP5QDE30) |
| Original GDN2 expand_v=2, 275M Cx8, LR 1.6e-3 | Did not reproduce: exact step36500 resume remained finite through step36780, crossing the old step36768 loss failure | step36500 | [01KYFWDJ18MZ20F88JA7CF52M4](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYFWDJ18MZ20F88JA7CF52M4) |
| GDN1 expand_v=2, 1.2B Cx8 | Did not reproduce in compact replay: finite through step17605; old 32-rank diagnostic had broad all-rank NaN gradients at step17592 | step17500 | [01KYFYV602MA89GK67TXM0WNYS](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYFYV602MA89GK67TXM0WNYS) |

All replays are read-only: checkpoint loading is enabled, but checkpoint
writes, pruning, W&B, and eval callbacks are disabled. The localizer checks
pre-reduction loss plus forward outputs, incoming backward gradients, and
module-created backward gradients. Captured recurrent layers are compared with
token-by-token FP32 references.

### Original GDN2 1.2B Cx4, step 9059

The exact replay reproduced the failure at the same step. All checkpoint
parameters were finite. The first non-finite boundary was rank 5,
`module.blocks.0.attention`, local sequence 1, at token 4992. Later forward
layers and the backward pass merely propagated those NaNs; the previously
observed 478/481 bad local gradients therefore did not originate in backward.

The trigger was an inactive/masked `finemath-3plus` slot containing only token
IDs 11 and 931, alternating perfectly for all 8192 tokens (4096 occurrences of
each; 8128/8128 lag-64 matches). This is more extreme than the repetitive
12-token canonical 480M failure. The exact reference comparison is
[01KYFXQ01DNQGGHBPT8YSE7RE0](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYFXQ01DNQGGHBPT8YSE7RE0).

The production chunk first became non-finite at token 4992; FLA's sequential
FP32 recurrence became non-finite at token 4980 in the same head/value channel
(head 11, channel 60). An FP64 replay of that exact state column remained
representable but reached `1.22e64`, crossing `1e38` at token 4890 and `1e50`
at token 6408. The homogeneous transition amplified the visited state on 4087
tokens (maximum one-step gain 1.042), matching the alternating input structure.
Across the long growth interval, the observed state magnitude compounds by
about 1.037 per two-token cycle (1.018 per token geometrically), which is
enough to turn ordinary values into FP32 overflow over thousands of repeats.
This independently reproduces the canonical finding: the optimized kernel is
accurately exposing an unstable learned recurrence, not creating the failure.

The instance filter had already marked this period-2 sample false, but the
current data path only masks its labels; the model still executes the sequence.
For a recurrent model, loss masking cannot protect the forward state from a
non-finite. Any data mitigation therefore needs to resample or replace filtered
input IDs before forward, not merely extend the filter and preserve the current
label-only mask behavior.

#### Actual FLA v0.5.2 release replay

The released `v0.5.2` tag is commit `9c8e42e`, not the older pinned
`cbb0a72` commit that also identifies its package version as `0.5.2`. A
[release-tag replay](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYJVPMPT2ZCP9XF5YB4FDZQX)
asserted the installed VCS commit and repeated this exact checkpoint, optimizer,
data position, 16-GPU/EP8/MB4 topology, and localization window. It reproduced
the same failure at step 9059: rank 5, block-0 GDN2 forward, local sequence 1,
token 4992. The actual release therefore does not change the established root
cause or mitigate this persistent recurrence overflow.

#### Original-checkpoint v0.5.2 release matrix

The release follow-up replayed all six original `expand_v=2`,
negative-eigenvalue GDN2 checkpoints that had repeated at the same
checkpoint-relative step. The canonical configuration is intentionally outside
this matrix. All jobs loaded the exact model, optimizer, and data state with the
original distributed topology, and disabled checkpoint writes, W&B, and evals.

| Original checkpoint | Historical failure | Actual `v0.5.2` result | Earliest non-finite |
|---|---:|---|---|
| 275M Cx8 `step36500` | 36768 | [Finite through 37000](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYJTMTS631KAFF2S7H10AEDH) | None |
| 810M Cx1 `step10000` | 10039 | [Exact recurrence](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYJXNYZR28BVW80Z5RYRRTK8) | Rank 6, block-10 attention forward, sequence 0/token 3072 |
| 810M Cx2 `step56500` | 56755 | [Exact recurrence](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYJXP1VWCX32JGJJ36R0QY73) | Rank 0, block-5 attention forward, sequence 1/token 5376 |
| 1.2B Cx1 `step8000` | 8029 | [Finite through 8045](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYJXP4X2HX91V3BYM8F6FMDW) | None |
| 1.2B Cx4 `step9000` | 9059 | [Exact recurrence](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYJVPMPT2ZCP9XF5YB4FDZQX) | Rank 5, block-0 attention forward, sequence 1/token 4992 |
| 1.2B Cx8 `step7000` | 7073 or 7125 | [Recurred at 7125](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYJXP82K2602JQCCVHSMY69C) | Rank 23, block-0 attention forward, sequence 1/token 7424 |

The Cx8 replay crossed its alternate step-7073 point but reproduced at its
other historical repeat point, step 7125. Its block-0 output contains NaNs from
token 7424 through the end of that local sequence, after which later blocks,
local loss, backward, and the optimizer become non-finite. Likewise, both 810M
replays first become non-finite in a GDN2 attention forward output. These three
new captures match the phase signature of the reference-verified 1.2B Cx4
failure, although token-by-token FP32/FP64 references have not yet been run on
the 810M and Cx8 captures.

The 275M and 1.2B Cx1 crossings mean those two historical failures are not
deterministic under the current source/runtime. Because the older source cannot
always be reconstructed exactly and an older-kernel 275M replay also crossed,
these crossings must not be interpreted as release fixes. Overall, four of six
original checkpoint-local failures reproduce with the actual release, so FLA
`v0.5.2` does not resolve the original GDN2 stability problem.

### Original GDN2 275M Cx8, old step 36768

This case did **not** reproduce. The read-only replay loaded exact `step36500`,
enabled all forward/backward/local-loss checks for steps 36750--36780, and
completed the hard-stop step with finite loss and gradients. No recurrent-layer
capture was emitted. The historical gradient-debug attempt did fail immediately
after step 36767 with non-finite loss on all eight ranks, so the old symptom is
real, but it is not deterministic from the saved checkpoint under the current
pinned source.

There is an important provenance limitation: the old full-sweep and first
gradient-debug jobs installed the pinned FLA GDN2 commit, but copied OLMo-core
from a mutable Weka checkout rather than recording a source Git SHA in their
Beaker specs. We therefore cannot reconstruct their exact OLMo-core code from
the job metadata. The non-reproduction rules out a permanently poisoned
checkpoint/optimizer state. It leaves a historical source difference or a
runtime/kernel/hardware-sensitive event as plausible explanations, but provides
no evidence that this particular event was the deterministic recurrent-growth
mechanism seen in the 480M and 1.2B cases.

### GDN1 1.2B Cx8, old step 17592

The historical 32-rank diagnostic definitely produced a different signature
from the deterministic GDN2 cases: at step 17592, 388--413 of 413 local
gradient entries were NaN on each rank, while the last completed step had
ordinary loss and gradients. It did not have module-phase hooks, however, so
the broad optimizer dump alone cannot prove whether the first non-finite was in
a GDN1 backward kernel or a forward value that only became visible during
backward.

The compact replay loaded exact `step17500`, passed a full finite-parameter
audit, and preserved the 96-sequence global batch on one EP=8 node by using
four MB3 accumulation microbatches. It completed steps 17580--17605 with finite
forward outputs, incoming gradients, module-created gradients, local loss, and
optimizer gradients. No capture was emitted. Thus this event is not a
deterministic function of the checkpoint and global sample set under the
current implementation.

This negative result has two limitations. It changes the original per-rank
microbatch grouping and DP reduction topology from 32 ranks/one microbatch to
8 ranks/four accumulated microbatches. More importantly, the historical
diagnostic pinned OLMo-core commit `45b2c821a`, whereas the replay uses the
post-migration code path; attention, DDP, optimizer, and MoE infrastructure all
changed substantially between those commits. The evidence therefore points to
a source/topology-sensitive or low-level nondeterministic numerical event, but
does not identify which one. It does rule out the persistent checkpoint poison
hypothesis and does not support assigning this GDN1 event to GDN2's proven
non-normal forward-state growth.

## Working mechanism

For normalized keys, GDN2's exact key-axis transition is

`A_t = (I - k_t (b_t * k_t)^T) Diag(exp(g_t))`.

The erase factor's nontrivial eigenvalue is bounded when `b` is bounded, but
`b_t * k_t` is generally not parallel to `k_t`. The full decayed transition is
therefore non-normal. Bounds on the erase eigenvalue do not bound its singular
values or a long product of changing, non-commuting transitions.
Repeated inputs can coherently excite the same state direction for thousands of
tokens, producing large transient amplification; the additive value write then
feeds the amplified direction. This remains possible with `expand_v=1` and
negative eigenvalues disabled, as the canonical failure demonstrates. Larger
value state and negative-eigenvalue support add degrees of freedom/opportunities
but are not necessary causes.

GDN1 is more constrained: setting the erase gate to a scalar beta makes
`b_t * k_t` parallel to `k_t`. With normalized keys and beta in [0, 2], the
ideal single-token delta update is a symmetric contraction before the bounded
additive write. KDA has the same stabilizing scalar-gate special case.
Consequently, if the GDN1 chunk kernel diverges but the FP32 sequential
recurrence stays finite, that would implicate numerical conditioning of the
parallel chunk/WY implementation (especially repeated nearly-collinear keys and
beta near 2), rather than the same mathematical instability as GDN2. The exact
GDN1 replay is intended to distinguish those mechanisms.

## Where the failures entered the experiment series

- SWA did not exhibit this recurring non-finite signature.
- The signature predates both GDN2 and element-wise attention gating. An
  ungated GDN1/NoPE 1.2B Cx8 trajectory failed at step 17592. A fresh
  reproduction later completed, so that older GDN1 event was not universally
  reproducible from initialization.
- Gated GDN1 also showed sporadic broad-gradient failures, but exact checkpoint
  replays crossed some formerly failing steps. Attention gating is therefore
  neither necessary nor sufficient for the failure.
- Original GDN2 (`expand_v=2`, negative eigenvalues enabled) made the failures
  much more frequent and persistently replayable.
- Canonical GDN2 (`expand_v=1`, negative eigenvalues disabled) reduced the
  frequency, and its complete 275M sweep was stable, but the exact 480M Cx2
  failure proves those settings do not remove the underlying GDN2 mechanism.
- The matched short-sequence kernel audit found no broad forward/backward
  disagreement: all comparisons passed FLA's combined tolerances, with cosine
  similarity above 0.99984. That is compatible with a recurrence that is
  ordinarily accurate but occasionally enters an extreme state trajectory on
  long, structured inputs.

## Conclusions and mitigations

The two persistently reproducible GDN2 cases share the same root mechanism:
long, highly periodic sequences coherently amplify a learned non-normal
recurrent state until FP32 overflow. The production chunk kernel and
token-by-token recurrence agree on the failing lane and nearly the same token;
FP64 only postpones overflow and reveals the enormous finite trajectory. These
are architectural state-stability failures exposed by data, not a general FLA
forward/backward correctness bug.

By contrast, neither the original 275M GDN2 event nor the representative GDN1
event reproduced under the pinned current path. The historical GDN1 signature
may still be a chunk-WY backward/conditioning problem--repeated nearly
collinear keys are a plausible stressor--but that remains a hypothesis. No
persistently reproducible backward-only failure has yet been demonstrated.

Recommended actions, in order:

1. Resample or replace filtered input sequences **before** model forward. The
   existing label-only instance mask cannot protect recurrent state. Extend
   repetition detection beyond trivial period 1, but do not treat filtering as
   a proof of architectural stability.
2. Keep synchronized non-finite step skipping as a training-continuity guard.
   It avoids poisoning optimizer state but cannot make the failed batch useful
   or repair the recurrence.
3. Evaluate a GDN2 stability constraint on a controlled 275M sweep: constrain
   transition singular gain, normalize/bound recurrent state or update norm,
   or reduce the channel-wise erase freedom toward the scalar-gate GDN1/KDA
   special case. `expand_v=1` and disabling negative eigenvalues reduce risk
   but do not eliminate it.
4. Treat GDN1 as a separate numerical investigation. If historical fidelity
   matters, port the phase hooks onto commit `45b2c821a` and replay with the
   original 32-rank/DP topology; compare the production chunk/WY backward with
   a sequential reference only if that exact setup reproduces.
5. Pin the exact OLMo-core commit in every future training and diagnostic job.
   Historical jobs that copied a mutable Weka checkout cannot be reconstructed
   precisely from their Beaker specs.
