# Bounded medium follow-up experiments

The locked model, BF16 compute, FP32 gradients/reductions/Adam, private EP outputs,
and WSD stay unchanged. All candidates are explicit, default-off profile variants.
They are not automatically promoted to production by adding them to this branch.

`optimized-lb-batched` adds `OLMO_PROFILE_LB_COUNT_BATCHED=1` and the independent
`OLMO_PROFILE_LB_COUNT_BATCHED_EP=1` authorization switch to the existing optimized
bundle. It preserves each router's original scores, logits, pre-drop local counts,
document counts and loss divisor. One packed FP32 count reduction per microbatch
replaces one reduction per layer. The existing per-layer loss and metric calculation
still runs once per record. No deferred replicated-gradient reduction is enabled.

Supported experimental scope: PP1/TP1, ordinary rowwise EP, no CP, TBO, activation
recomputation, shared EP outputs or FP8. Records are call-local; no mutable
cross-forward accumulator or shared dispatch-buffer ownership is introduced.
The separate EP opt-in is mandatory: setting the older batching flag alone still
rejects EP. The no-EP path retains its existing behavior.

Qualification must precede a full-model timing run:

- Existing primitive and actual no-EP two-layer regression gates.
- New two-layer rowwise EP4 and EP8 tests, eager and compiled, eight accumulated
  microbatches and three Adam updates. Compare CE, gradients, parameters, every
  optimizer state, exact global counts, and auxiliary metrics; require no drops.
- The EP8 fixture uses medium expert dimensions, but ordinary attention and a
  short sequence. It is not a numerical sign-off for the entire KDA model.
- Full production-shape matched timing, routing/drop telemetry, and a fresh trace
  must verify a net gain and the intended reduction in collective count.
- Exact-topology synchronous save/resume remains required for any CBS selection.

At introduction, only CPU planning checks and syntax/format checks have run.
Do not infer a successful GPU qualification from this document.
