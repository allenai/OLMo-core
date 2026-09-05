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

## Completed profile and bottlenecks

- Data loading is not a substantial bottleneck: baseline mean 2.71ms per 3.36s update.
- Baseline per-step rank-reduced memory: 175.466 GiB active / 178.563 GiB reserved.
  End-of-run allocator counters are reset every step and must not be called full-run peaks.
- The all-rank Nsight attempt segfaulted at capture start. Pre-capture timing survived,
  but the surviving report contains no CUDA trace data. No kernel attribution is claimed
  from that failed capture.
- [Independent PyTorch capture](https://beaker.org/ex/01M1QNHVHXXPSJHBFF7KBB7SCV):
  completed successfully on all 64 GPUs. Eight representative ranks (one per node)
  have Chrome traces and distributed-event summaries; rank0 has an allocator snapshot.
  Raw artifacts total approximately 176 MiB and remain on Weka, not Beaker results.

The two-update rank0 trace contains 67,548 GPU kernels over a 9.047s span.
Kernel-busy union is 8.855s (97.9%); the 191ms without kernels is not necessarily idle,
because memory operations can occupy some of it. Other sampled ranks show the same
kernel count and approximately 9.04s span. This is not predominantly a data-feed stall.

| Work observed in rank0's two-update capture | GPU duration sum | Interpretation / next action |
|---|---:|---|
| NCCL collectives | 3.192s | 2.659s collective-only union, with no other kernel running; investigate overlap/payloads, but do not equate this to removable time. The controlled reduce-scatter test was a tie. |
| Grouped expert GEMMs (`aten::_grouped_mm`, profiler table) | 1.117s | Real compute remains material. Narrow expert matrices and packing deserve targeted GEMM/epilogue work; changing expert counts or widths would change the model and is excluded. |
| Matrix multiplies (`aten::mm`, profiler table) | 1.149s | Separate from grouped expert matmuls; operator totals are not an additive step-time decomposition. |
| FP32 add elementwise kernel | 0.671s | Consistent with FP32 gradient accumulation and tensor traffic. Direct gradient accumulation in GEMM epilogues is a possible follow-up, with reduction order and correctness checks; not implemented here. |
| Fused EMO mask/sort/indexing kernel | 0.589s | The second argsort is unnecessary permutation inversion; exact inverse-scatter alternative is qualified and in full-model A/B. |
| Fused expert activation/backward kernel | 0.444s | Another substantial memory/elementwise component; preserve mathematical and precision semantics in any fusion experiment. |
| KDA backward intra kernel | 0.360s | Default dispatch still uses FLA at this shape. The independently qualified cutoff128 variant already improves unprofiled end-to-end TPS by 5.03%. |

These durations overlap and must not be added. CPU-side `Command Buffer Full` events
are queue backpressure, not evidence by themselves that the CPU is starving the GPU.
The profile run's uncaptured windows were approximately 58.8k TPS/GPU, versus 76.9–78.2k
in independent unprofiled baselines. Instrumentation/compiler effects and node differences
are not separated, so the profile identifies candidate work, not a clean speedup budget.
The kernel names alone also do not establish which NCCL protocol was selected.

The [read-only diagnostic export](https://beaker.org/ex/01M1QQYF7BVWQ6RB9WAW8XZ61F)
confirms each update has approximately **46.55 GiB of logical FP32 gradient all-reduce
input**, plus a BF16 parameter all-gather with **23.28 GiB full output** (0.364 GiB
input per rank). These are tensor payloads, not measured network-wire bytes. There are
also 120 small 512-element FP32 all-reduces per update, matching 15 MoE layers ×
8 microbatches of global load-balancing statistics. The optimizer-sharded reduction
experiment includes ordinary BF16 expert parameters too; "normal" excludes FP8 stores,
not experts. No further collective change is made from these counts alone.

Rank0's end-of-capture snapshot has 72.06 GiB allocated and 178.33 GiB reserved.
The 100,000-event history is full/truncated, so it cannot establish a full-run peak.
Its largest cumulative allocation sites are expert activation/backward and grouped
GEMM buffers, consistent with the transient-memory bottleneck. Cumulative allocations
are traffic, not simultaneous live memory. All raw captures and source checkpoints
remain untouched; the diagnostic job used no GPUs.

## Other controlled candidates

- [Compiler-safe disabled NVTX A/B](https://beaker.org/ex/01M1QNHSC6ZVAGBGV38KAGW605): completed.
  A local reproducer shows that the current custom no-op context prevents Dynamo
  resuming around opaque calls. The alternative is opt-in and has CPU correctness tests.
  On its own same-node allocation, median TPS/GPU was 76,921 baseline vs 77,276
  alternative (+0.46%); means 76,857 vs 77,260 (+0.52%). Both completed 60 updates
  with finite losses and no skips. This is a small single-pair difference, not a robust
  large gain; the flag remains off by default. Do not compare its absolute TPS directly
  against the earlier matrix on different nodes. Mean CE was 1.950748 vs 1.950960.
- [EMO pool-mask microbenchmark](https://beaker.org/ex/01M1QQ1MGM97EYNDDTZ379CPWE): passed.
  EMO's second argsort only inverts a permutation. Inverse scatter preserves the exact
  first-sort tie policy and removes that redundant sort. Six CPU tests passed.
  On B300 at 4×8192×512 scores, median compiled mask time fell from 1.7313ms to
  1.1354ms (-34.4%); eager time fell from 1.9504ms to 0.8453ms (-56.7%).
  Masks were exact against the baseline within each execution mode, including ties.
  The first qualification attempt incorrectly compared compiled results against eager
  results on tied scores; even the unchanged baseline differs across those modes because
  the argsort is not stable. The corrected test compares like execution modes.
- [Full-model EMO A/B](https://beaker.org/ex/01M1QQCYKN4AGBMRHX84618N6C): launched.
  Same eight nodes, independent step7500 restores: baseline, inverse-scatter alone,
  inverse-scatter plus KDA cutoff128. Source `680ed4ae2`. No model/batch/precision changes.
  The mask microbenchmark improvement is not yet an end-to-end improvement claim.

## PR859 audit: already-covered kernel implementation

Reviewed [PR859](https://github.com/allenai/OLMo-core/pull/859) at
`fe0195dd845a3dd786e4fc80d52eed9e7ce404b0` on 2026-09-05.
Its vendored `VENDORED_FROM` is `7a6983baf2beb4ec4d7fe914ec9f6670438af99b`,
the exact package revision installed by this profiling branch.
Git blob comparison confirms all **23 files** under `_common/` (4), `kda/` (14),
and `cconv/` (5) are byte-identical, with no missing or extra files in those families.
Training call sites have the same arguments and opt-in behavior for KDA and all three
short convolutions; the relevant wiring difference is the import namespace and removal
of the installed-package availability check. The branch also has unrelated ancestry and
inference changes, which are not required for this training experiment.
Live timing-matrix logs confirm `kernel-fun cconv: engaged` for both D=1024 and D=2048,
at B4/T8192/W4 with BF16 inputs and BF16 weights. This is not merely an installed but
unused optimization.

Therefore PR859 is a vendored distribution of the kernels already under test, not a
fourth kernel-performance arm. No duplicate kernel tree was merged and no duplicate
64-GPU job was launched. The default CTA256 gate is identical too: our CTA128 trial
remains a separate, explicitly labeled profiling override, not PR859's default behavior.

The author's prospective further KDA fusion is not code in the reviewed PR. It could
reduce intermediate memory traffic and launch count without changing architecture,
but the reported 7% gain on a 275M ladder model is not a measured gain on this model,
and a prospective 30% layer improvement is not 30% end-to-end TPS. We already have
real traces to inform that work. Preserve expand_v=2, negative eigenvalues, FP32 gate/
state semantics, gradients, and the existing no-CUDA-graph-capture safety fallback.
Assess the full KDA forward/backward region including projections and epilogues rather
than summing overlapping kernel scopes, and validate with the same clean timing window.
The user clarified that the author's **original ~30% regression referred to overall
training TPS when replacing attention with KDA**, not an isolated layer benchmark.
That historical comparison is distinct from his prospective further-fusion estimate.
Further fusion is deferred until the current controlled comparisons are settled.

## The 600-TFLOPs target

With this FLOP accounting, 600 TFLOPs/GPU requires about 137.65k TPS/GPU and
1.904s/update. That is 75.97% above the baseline or 67.54% above the cutoff128 result.
The observed KDA improvement alone is not close. Larger gains need measured reductions
inside each microbatch, not just an optimizer communication flag. Kernel-duration sums
must not be added across overlapping streams and treated as end-to-end time savings.

All experimental changes remain isolated in `codex/small-16mi-profile-v2`.
