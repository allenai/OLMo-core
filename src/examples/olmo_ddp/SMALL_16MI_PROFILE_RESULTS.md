# Small 16Mi production-candidate profiling — results in progress

Updated 2026-09-05 UTC. See [protocol and run ledger](SMALL_16MI_DEEP_PROFILE.md)
for source revisions, exact model settings, checkpoint migration, and failed attempts.
No production/CBS checkpoint or uploader state is modified by these tests.

## Controlled same-node comparisons

All rows below: eight Holmes nodes / 64 B300s, 16,777,216 tokens/update,
MB4 × 8192 tokens, GA8, PP1/EP1/DP64, BF16, EMO, per-head QK gains from PR855.
Each arm independently restores trained step7500, warms up, and runs 60 updates.
Statistics use only updates31–60. No CUPTI profiler is active in these timing arms.
FLOP accounting is 4,358,934,624 idealized training FLOPs/token throughout.

| Implementation | Mean TPS/GPU | Median TPS/GPU | Median TFLOPs/GPU | Median step (s) | Median TPS change | Mean CE | Skipped steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline: current kernel-fun default dispatch | 78,081 | 78,222 | 340.965 | 3.3513 | — | 1.951015 | 0 |
| Normal-gradient reduce-scatter | 78,114 | 78,180 | 340.780 | 3.3531 | -0.05% | 1.950893 | 0 |
| KDA CTA cutoff128 | 82,049 | 82,159 | 358.125 | 3.1907 | +5.03% | 1.950877 | 0 |

[Completed timing matrix](https://beaker.org/ex/01M1QM9ZVJG0CB5B2609FW77SX).
All eight jobs exited successfully after all three arms.

Reduce-scatter is effectively tied; there is no speed reason to change that default.
KDA cutoff128 is a measured short-run improvement at this exact shape. The package's
default cutoff256 falls back to FLA for our 128-CTA grid; the alternative engages the
new CuTe chain. It does not change model dimensions, precision, batch, or LR.

KDA numerical checks: first-update CE differs from baseline by -0.000024915,
and first-update total gradient norm by +0.01885%. The independent one-GPU layer
qualification also checked output, input gradient, and all parameter gradients.
These checks and 60 finite updates do not establish long-run stability or eval parity.
The tiny mean-CE differences in this table are not evidence of a quality improvement.

## Profile and other candidates

- Data loading is not a substantial bottleneck: baseline mean 2.71ms per 3.36s update.
- Baseline per-step rank-reduced memory: 175.466 GiB active / 178.563 GiB reserved.
  End-of-run allocator counters are reset every step and must not be called full-run peaks.
- The all-rank Nsight attempt segfaulted at capture start. Pre-capture timing survived,
  but the surviving report contains no CUDA trace data. No kernel attribution is claimed
  from that failed capture.
- [Independent PyTorch capture](https://beaker.org/ex/01M1QNHVHXXPSJHBFF7KBB7SCV): pending.
- [Compiler-safe disabled NVTX A/B](https://beaker.org/ex/01M1QNHSC6ZVAGBGV38KAGW605): pending.
  A local reproducer shows that the current custom no-op context prevents Dynamo
  resuming around opaque calls. The alternative is opt-in and has CPU correctness tests.
- [EMO pool-mask microbenchmark](https://beaker.org/ex/01M1QP9BKYNWPE0QVEK34MX8BN): pending.
  EMO's second argsort only inverts a permutation. Inverse scatter preserves the exact
  first-sort tie policy and removes that redundant sort. Six CPU tests passed; GPU timing
  and an end-to-end comparison are still required.

## The 600-TFLOPs target

With this FLOP accounting, 600 TFLOPs/GPU requires about 137.65k TPS/GPU and
1.904s/update. That is 75.97% above the baseline or 67.54% above the cutoff128 result.
The observed KDA improvement alone is not close. Larger gains need measured reductions
inside each microbatch, not just an optimizer communication flag. Kernel-duration sums
must not be added across overlapping streams and treated as end-to-end time savings.

All experimental changes remain isolated in `codex/small-16mi-profile-v2`.
