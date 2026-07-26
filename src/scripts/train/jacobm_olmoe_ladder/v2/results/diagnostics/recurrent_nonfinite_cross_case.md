# Cross-case GDN non-finite analysis

Status: in progress (2026-07-26)

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
| Original GDN2 expand_v=2, 275M Cx8, LR 1.6e-3 | Step 36768; non-finite loss on every rank | step36500 | [01KYFWDJ18MZ20F88JA7CF52M4](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYFWDJ18MZ20F88JA7CF52M4) |
| GDN1 expand_v=2, 1.2B Cx8 | Step 17592; broad all-rank NaN gradients | step17500 | [01KYFWFR4Z1EJSNAQNX1HTDDG0](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYFWFR4Z1EJSNAQNX1HTDDG0) |

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
This independently reproduces the canonical finding: the optimized kernel is
accurately exposing an unstable learned recurrence, not creating the failure.

The instance filter had already marked this period-2 sample false, but the
current data path only masks its labels; the model still executes the sequence.
For a recurrent model, loss masking cannot protect the forward state from a
non-finite. Any data mitigation therefore needs to resample or replace filtered
input IDs before forward, not merely extend the filter and preserve the current
label-only mask behavior.

## Working mechanism

For normalized keys, GDN2's exact key-axis transition is

`A_t = (I - k_t (b_t * k_t)^T) Diag(exp(g_t))`.

Its nontrivial eigenvalue is bounded when `b` is bounded, but `b_t * k_t` is
generally not parallel to `k_t`. The transition is therefore non-normal. Bounds
on one transition's eigenvalues do not bound its singular values or a long
product of changing, non-commuting transitions.
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

## Pending results

- First failing rank, block, phase, sequence, token, head, and channel for each replay.
- Whether FP32 sequential recurrence also diverges.
- Repetition/periodicity statistics for each exact failing sequence.
- Cross-case mitigation recommendations.
