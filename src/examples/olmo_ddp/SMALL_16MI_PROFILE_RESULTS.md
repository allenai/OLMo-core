# Small 16Mi production-candidate profiling — results in progress

Updated 2026-09-05 UTC. See [protocol and run ledger](SMALL_16MI_DEEP_PROFILE.md)
for source revisions, exact model settings, checkpoint migration, and failed attempts.
No production/CBS checkpoint or uploader state is modified by these tests.

**Current status (06:03 UTC):** Nsight Systems is fixed and validated on the full
64-B300 model under standalone 2026.4.1. The latest healthy same-node comparison is
78,115 → 82,190 → 85,888 median TPS/GPU for baseline → KDA128 → KDA128+EMO inverse
scatter (+9.95%, 374.38 idealized TFLOPs/GPU). The default-off gradient-add candidate
passed exact two-GPU sharded-Adam qualification and improved its subsequent same-node
A/B from 84,523 to 88,234 median TPS/GPU (+4.39%, 384.61 idealized TFLOPs/GPU).
The faster candidate is running fresh timing/PyTorch/light-Nsight passes. The next
paired-SwiGLU backward candidate passed exact one-GPU and two-GPU compiled/sharded-Adam
checks; its 64-GPU same-allocation A/B is submitted, with no full-model gain claimed yet.
See the dated healthy-confirmation and Nsight sections below; the earlier bad-fabric
capture is retained as diagnostic history, not the current performance reference.

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

**Topology warning (2026-09-05 03:24 UTC):** the capture below and the first EMO
matrix both used a degraded node. Their communication timings are not representative
of a healthy eight-node allocation. See the confirmed NVLink findings below.

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
in independent unprofiled baselines. The later unprofiled EMO baseline also ran near
58.8k on the same degraded topology. This is not explained by profiler overhead alone;
the trace identifies candidate work, not a clean healthy-cluster speedup budget.
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

## Baseline reproducibility: disconnected NVLink GPU

Read-only host inspection on 2026-09-05 identified **GPU1 on
`holmes-cs-aus-485.reviz.ai2.in`**, UUID
`GPU-abfa9313-09fd-6bc9-ca4d-7bedd9019cef`, with all NVLink links inactive.
`nvidia-smi topo -m` shows GPU1 reaching the other seven GPUs through NODE/SYS,
whereas their mutual links show NV18. `nvidia-smi nvlink --status -i 1` reports:
`NVML: Unable to retrieve Nvlink information as all links are inActive`.

The current EMO matrix's node4 job is `01M1QQCZNWS27BTP1E87PRHKQP`. Its NCCL logs
place global rank33 / GPU1 alone (`localRanks 1`) and its seven peers together
(`localRanks 7`). Both slow runs report `nRanks 64 nNodes 9`, despite eight physical
nodes. Healthy timing allocations report eight NCCL nodes. All use the same image,
NCCL 2.28.9+cuda13.0, model, batch, and relevant launch flags.

| Allocation | Median/observed baseline TPS/GPU | NCCL nodes | Collective channels | NVLS channels |
|---|---:|---:|---:|---:|
| Original timing matrix | 78,222 median | 8 | 28 | 32 |
| Compiler-no-op A/B | 76,921 median | 8 | 16 | 32 |
| PyTorch capture | ~58,800, uncaptured windows | 9 | 4 | 0 |
| First EMO matrix | 58,832 median | 9 | 4 | 0 |

This is a concrete hardware/fabric degradation and the leading explanation for the
slow tier, not a new model/compiler regression. A fresh healthy-node comparison is
still required to quantify recovery and isolate any remaining instrumentation effects.
Do not force NCCL to assume connectivity that the hardware does not provide.

Follow-ups exclude host485 via the existing `OLMOE3_ALLOWED_HOSTNAMES` allowlist.
`olmoe3_profile_topology.py` validates the entire eight-GPU local NVLink matrix before
workers launch, retaining diagnostics on Weka. No GPU reset, node cordon, or other
administrator action was attempted. Infrastructure needs to inspect/repair this host.

From 06:02 UTC, new allocations also exclude host516, which failed Beaker's pre-start
health check twice (initial tasks in the gradient-add A/B and its subsequent profile).
Beaker replaced both before training started. This is not evidence of the same NVLink
fault as host485; no host-level intervention was performed.

## Non-fusion communication experiments

- Added opt-in `OLMO_PROFILE_RS_SINGLE_PARAM_FAST_PATH=1`: a single-parameter
  gradient bucket is already in rank-major order, so it skips scratch packing and
  copy-back. Dtype conversion still uses a dedicated per-bucket communication buffer.
  Multi-parameter buckets keep the original permutation and stable asynchronous input
  ownership. The production/default flag remains off.
- Nine new CPU/gloo distributed tests passed (three layout/group patterns × three
  precision modes, both old/new paths, repeated reductions). Three existing gloo
  gradient/accumulation/group-routing tests also passed with the flag enabled; two
  GPU-only cases were skipped locally. Eleven topology/sequential-plan tests passed.
- GPU qualification runs the new NCCL parity tests, existing optimizer/gradient tests,
  and an isolated 1/2-GiB gradient-bucket microbenchmark in old/new/old order on two B300s.
  Isolated timing must not be reported as a 64-GPU whole-model gain.
- Optimizer gathering remains investigation-only. It gathers into rank-major temporary
  storage, then copies into parameter-major contiguous model views. Simply gathering
  directly into the model buffer would change layout; simply retaining the 23.28-GiB
  temporary can increase live memory despite allocator caching. No such change is made
  without new lifetime/layout tests and evidence from a healthy-node profile.

Two-B300 GPU qualification completed successfully (exit0). Nine new NCCL parity
cases and three existing gradient/optimizer cases passed; two tests requiring four
GPUs were deliberately skipped. All six isolated benchmark arms checked their exact
reduced gradients. Times include gradient scaling and reduce-scatter, but exclude
initialization, input filling, and the inter-iteration barrier; five warmups and
20 timed iterations per arm, using the slower rank's CUDA-event duration each time.

| FP32 gradient bucket | Old packing, before (median ms) | Single-param path (median ms) | Old packing, after (median ms) | Latency reduction vs mean of old medians |
|---|---:|---:|---:|---:|
| 1 GiB | 2.1572 | 1.4609 | 2.1507 | 32.18% |
| 2 GiB | 4.1583 | 2.8231 | 4.1491 | 32.03% |

This is **not a 32% training speedup**. A crude serial sum of the saved milliseconds
over 15 one-GiB and 15 two-GiB expert buckets is about 30ms/update, before considering
different 64-rank collective algorithms and overlap. It motivates a narrow follow-up,
not changing the production default or prioritizing another 64-GPU allocation before
the healthy-node profile. Source `92a24ffa0`; result dataset
`01M1QT7ZYYQY7EXGNFZAG642PM`. Kernel-fun is not used by this isolated DDP operation;
the full-model comparison separately verifies its pinned package during setup.

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
- [Full-model EMO A/B](https://beaker.org/ex/01M1QQCYKN4AGBMRHX84618N6C): completed,
  all eight jobs exit0.
  Same eight nodes, independent step7500 restores: baseline, inverse-scatter alone,
  inverse-scatter plus KDA cutoff128. Source `680ed4ae2`. No model/batch/precision changes.
  All three arms completed 60 finite-loss updates with zero skipped optimizer steps.
  The allocation included the disconnected-NVLink host described above: these are
  controlled **degraded-topology** improvements, not healthy production throughput.

| First EMO matrix arm (degraded topology) | Mean TPS/GPU | Median TPS/GPU | Change vs its baseline | Median step (s) | Mean CE |
|---|---:|---:|---:|---:|---:|
| Baseline | 58,781 | 58,832 | — | 4.4558 | 1.951032 |
| EMO inverse scatter | 60,950 | 60,991 | +3.67% | 4.2981 | 1.950536 |
| KDA cutoff128 + EMO inverse scatter | 62,944 | 63,007 | +7.10% | 4.1605 | 1.950713 |

Same updates31–60 and FLOP accounting as the healthy timing table. First-update CE
is 1.98963308 baseline, 1.98964000 EMO, and 1.98960662 combined; gradient norms
0.07341947, 0.07342064, and 0.07343025. This supports short-run numerical agreement,
not long-run equivalence or a quality ranking. Do not combine this +7.10% with the
earlier healthy KDA-only +5.03% as though they were additive gains.

## Follow-up jobs launched 2026-09-05

- [Healthy-node confirmation and capture](https://beaker.org/ex/01M1QT33FG4X90DBC4WE44ZP3V),
  source `5c7d46b4d`: one eight-node / 64-B300 allocation, urgent, allocated with
  1h minimum runtime. Same-node baseline, KDA-only, and KDA+EMO unprofiled timings,
  then a short KDA+EMO PyTorch capture. Host485 excluded; strict full-node NVLink
  preflight. Model, precision, checkpoint, batch, LR, and parallelism unchanged.
- [Two-GPU reduce-scatter qualification r2](https://beaker.org/ex/01M1QT7ZYNN58QK72YMY2CKDXK),
  source `92a24ffa0`, urgent/allocated10m: completed exit0 at 03:43 UTC, with all
  12 selected NCCL/gradient/optimizer tests passing and both bucket sizes faster.
  r1 (`01M1QT0W0CRGTESJNZ7M0KTNHT`) stopped at the guard because its container exposed
  two GPUs, not eight. Fixed by an explicit `--gpus 2` option; the 64-GPU guard remains
  strict. No training/model failure occurred in r1 and it is not still running.
- [Automatic CPU-only collector](https://beaker.org/ex/01M1QT82PQS02QRDBM3KDSF63X),
  source `92a24ffa0`, Rhea, urgent/unallocated0s, waiting for all four healthy passes.
  Only small summaries become Beaker results; raw traces remain on Weka.

At 03:44 UTC all eight healthy-confirmation jobs had started; the collector was
running and waiting for completed passes. No throughput result is claimed yet.

At 03:47 UTC the healthy run's NCCL initialization confirmed `nRanks64 nNodes8`,
28 collective channels and 32 NVLS channels. All eight strict local topology
preflights passed. At 03:57 the baseline was still in cold compilation, so there
was not yet a fresh steady-state TPS measurement. The infrastructure team has been
alerted to host485 by the user.

**Fresh healthy timings completed 04:22 UTC:** all three 60-update arms finished and
all 64 memory completion markers per arm were collected. Unprofiled updates31–60:

| Healthy confirmation arm | Mean TPS/GPU | Median TPS/GPU | Median TFLOPs/GPU | Median step (s) | Mean CE | Window skips |
|---|---:|---:|---:|---:|---:|---:|
| Baseline, default KDA dispatch | 78,092 | 78,115 | 340.500 | 3.3559 | 1.950790 | 0 |
| KDA cutoff128 | 82,139 | 82,190 | 358.261 | 3.1895 | 1.950825 | 0 |
| KDA cutoff128 + EMO inverse scatter | 85,831 | 85,888 | 374.381 | 3.0522 | 1.950823 | 0 |

Baseline TPS standard deviation is 289 across the 30 timed updates. Median cluster
throughput is approximately 5.00M tokens/s. This is within 0.14% of the original
healthy 78,222 median, and 32.8% faster than the degraded 58,832 median. Different
allocations were used for the hardware comparison, so this is strong corroboration
of the confirmed NVLink fault rather than a perfectly isolated one-node swap test.
First-update CE is 1.9896239042; gradient norm 0.0734193027. Active/reserved memory
in the clean window is 175.466 / 178.504 GiB. KDA-only improves median TPS by 5.22%;
the combination improves it by 9.95% over baseline and 4.50% over KDA-only. Combined
median cluster throughput is 5.497M tokens/s. TPS standard deviations are 289 / 295 /
357 respectively, across the 30 measured updates. Combined first-update CE is
1.9896166325, gradient norm 0.0734279081; active/reserved memory is 175.465 / 175.799 GiB.
Loss agreement is a short-run numerical check, not a long-run stability guarantee.
All three full 60-update metric files have now been checked: no skipped updates and
finite CE throughout. The combined PyTorch capture and collector completed at 04:33 UTC,
with all eight trace files present. Do not use profiled step timings in the unprofiled
comparison above. Compact dataset `01M1QT82PW9T4YB5G1FMFRW7FX` is downloaded locally.

### Healthy combined-candidate trace

The two-update rank0 trace spans 6.623s, contains 68,956 kernels, and has a 6.081s
kernel-busy union. Collective union is 1.053s; collective-only union is 0.492s.
Across the eight representative ranks, collective-only union ranges 0.492–0.909s
over the two updates. This is dramatically below the earlier degraded-node capture,
but different kernel/EMO variants were also used, so it is not a one-variable fabric A/B.
Do not equate collective-only union with a removable critical-path budget.

| Healthy rank0 work, two updates | GPU duration sum | Immediate use |
|---|---:|---|
| `aten::mm` | 1.189s | Recover input shapes and distinguish BF16 tensor-core matmuls from FP32 SIMT matmuls. |
| `aten::_grouped_mm` | 1.126s | Counter-profile isolated expert GEMM; preserve trained-routing caveat for synthetic inputs. |
| NCCL | 1.053s | Inspect overlap/launch timing in repaired Nsight; do not prioritize from old bad-node fractions. |
| `aten::add_` | 0.727s | Dominant 0.680s mixed-type FP32 accumulation kernel; test contiguous vectorized addition independently. |
| KDA backward | 0.545s | Improved dispatch still leaves material work; counter-profile hot kernels before further fusion. |
| Expert activation backward | 0.456s | Candidate for later fusion, after memory-traffic and correctness analysis. |
| Remaining EMO sort/mask core | 0.222s | Inverse permutation is already optimized; the first sort still does real work. |
| KDA forward | 0.184s | Preserve FP32-sensitive paths and current numerical behavior. |

These are nonadditive attribution sums, not a pie chart of step time. Uncaptured windows
in this pass return to 85,674 and 86,063 median TPS/GPU, agreeing with the independent
85,888 timing arm. The updated diagnostic export includes hot-op shapes/dtypes/strides
when present in the PyTorch trace. A one-GPU probe qualifies Nsight Compute counter
access and benchmarks a narrow vectorized gradient-add prototype with FP32 arithmetic;
it is **not enabled in production**. Synthetic uniform expert routing is explicitly
labeled and will not be reported as trained-routing performance.

The [shape/stride diagnostic](https://beaker.org/ex/01M1QXT5KQA29CW7RBSGKM1QYM)
completed at 04:40 UTC. It confirms the two large gradient adds are contiguous in
both arguments: FP32 destination / BF16 source, `[512,2048,512]` and `[512,1024,512]`.
Together they sum to 0.627s over two updates (240 calls of each shape). Router matrix
multiplies are FP32, with `[32768,1024] × [1024,512]` forward and corresponding backward
shapes. Those three FP32 GEMM shapes sum to 0.454s over the capture. Precision has not
been changed to accelerate them.

### Vectorized gradient-add qualification

[One-B300 probe](https://beaker.org/ex/01M1QXWSA88MHCEG947BFK6VZS), source `c21fade64`:
the standalone numerical/memory-bandwidth benchmark passed. Hardware-counter collection
then failed with `ERR_NVGPUCTRPERM` for native add, Triton add, and grouped GEMM. All three
workloads themselves completed with finite outputs. This is a Beaker/host counter-access
restriction, not a kernel crash. No host privileges, clock locks, or driver settings
were changed. Nsight Systems does not require these hardware counters.

| Gradient shape | Native before/after (median ms) | Triton block2048/4warps (median ms) | Isolated speedup | Logical traffic rate, native → Triton |
|---|---:|---:|---:|---:|
| `[512,1024,512]` | 0.83293 / 0.83299 | 0.39370 | 2.12× | 3.22 → 6.82 TB/s |
| `[512,2048,512]` | 1.65840 / 1.65842 | 0.77738 | 2.13× | 3.24 → 6.91 TB/s |

Five warmups and 20 CUDA-event timed calls per arm, native / three Triton launch settings /
native. Every arm matched after 25 additions with zero numerical tolerance. Separate
checks covered empty/tail tensors, repeated additions, infinities, NaNs, and large values.
The other two block/warp settings were within about 0.5% of block2048; no broad tuning
sweep is warranted. Logical traffic is 10 bytes/element (FP32 read/write + BF16 read),
not measured HBM-counter traffic. Dataset `01M1QXWSAFH471B2YQFGKF92AG` retains both the
successful benchmark and the counter-access failure.

Serial extrapolation at 120 pairs/update suggests about 158ms saved, before overlap,
interference, and hook overhead. This is not a 2× training speedup. The kernel is now
available behind default-off `OLMO_PROFILE_FP32_GRAD_ADD_VECTORIZE=1`, only for contiguous
CUDA BF16-to-FP32 additions of at least 64Mi elements; other inputs keep the native path.
[Two-GPU distributed qualification](https://beaker.org/ex/01M1QYBC15BSMXYRV476B450A7),
source `1c3d41fac`, completed exit0 at 04:54 UTC. Eight microbatches × three Adam
updates matched reduced gradients, model weights, and optimizer states with zero
numerical tolerance, including ordinary zeroing and buffer rebinding. The test checked
24 calls through the actual >=64Mi-element fast path; four existing NCCL gradient/
no-sync tests also passed. Local CPU fallback passed separately. Dataset
`01M1QYBC1EBXV2CBH3Y3HWTE2E` retains the test outputs.

The [64-GPU timing A/B](https://beaker.org/ex/01M1QYS6RMFNV0BP28QXW16W4A) completed
at the same source: KDA128+EMO inverse scatter, then the identical setup with vectorized
gradient addition. Each independently restores step7500 for 60 updates; compare only
updates31–60, not compilation or tracing. Host485 is excluded and all nodes must pass
NVLink preflight. [CPU-only result collector](https://beaker.org/ex/01M1QYT2B1XGS0QPMQ1R901X9N)
completed exit0 at 05:31 UTC, dataset `01M1QYT2B7DDJF9AR8HYY70CDH`. All eight training
tasks completed successfully. Beaker replaced a node after a pre-start health-check
failure; no replacement occurred between arms. All nodes passed mandatory NVLink preflight.

| Same-allocation arm, updates31–60 | Mean TPS/GPU | Median TPS/GPU | Median TFLOPs/GPU | Median step (s) | Mean CE |
|---|---:|---:|---:|---:|---:|
| KDA128 + EMO inverse scatter | 84,408 | 84,523 | 368.43 | 3.10145 | 1.951235 |
| + vectorized FP32 gradient add | 88,129 | 88,234 | 384.61 | 2.97100 | 1.950901 |

The measured gain is **4.39% TPS**, saving **130.44ms/update**, consistent with the
isolated ~158ms serial estimate after overlap/overhead. TPS standard deviations are
365/436; the timing sample ranges do not overlap. Both arms use exactly the same
provenance except variant name and gradient-add flag. Active memory is unchanged at
175.465GiB; reserved memory is 178.361/175.799GiB (allocator behavior, not a claimed
reduction in required model memory). All 60 updates per arm have finite CE/gradient
norms and zero skipped optimizer steps.

First-update CE difference is -1.19e-7 and gradient-norm difference +2.10e-6. Across all
60 updates, mean CE difference is -0.000178; maximum absolute per-step CE difference
is 0.006344 at step7555, whose gradient norm also differs by 0.0450. Therefore the full
trajectories are **not bitwise equivalent**, despite exact isolated/distributed add
checks. No long-run quality/stability claim or production default change is made.
The earlier 85,888-TPS combined arm ran on a different allocation; use 84,523 here
for this optimization's controlled comparison.

This wave includes fresh timing/torch/light-Nsight passes of the improved candidate,
and the completed one-B300 probe of compiled expert SwiGLU at `[524288,2048]` described
below. That probe compares default / pointwise-coordinate tuning / default with exact
forward/backward checks and saved generated code. The lighter
Nsight mode disables per-op autograd NVTX only; CUDA/NCCL and application NVTX remain.

Follow-up [64-GPU timing/torch/light-Nsight run](https://beaker.org/ex/01M1R17MJ5BNM7QS6NPVTBJQY8),
source `a401daccf`, is running with [CPU collector](https://beaker.org/ex/01M1R17XVBFB9MNDG4MRX93Y28).
It uses the gradient-add candidate, three independent 60-update restores, and
`OLMOE3_NSYS_AUTOGRAD_NVTX=0`. The revised callback passed lifecycle tests with and
without autograd annotations. This run does not contain the new activation prototype.

### Next target: expert activation backward

[One-B300 compiler probe](https://beaker.org/ex/01M1R18061X37H5332NRN9EMCZ), source
`a401daccf`, completed exit0 at 05:40 UTC. On synthetic `[524288,2048]` BF16 expert
inputs, default / pointwise-coordinate-tuned / default forward+backward medians were
2.26384 / 2.16994 / 2.26381ms. All outputs and gradients matched exactly. Tuning alone
saves ~0.094ms per call, about 11ms/update at 120 calls; no global compiler flag changed.
Dataset `01M1R180678E07Z4NVFGG4BANZ` includes generated forward/backward source.

The generated backward indexes the concatenated output, separately computes both
halves, and repeats loads/exponentials. Its intermediates remain FP32; the actual
generated code (not just its annotated graph) rounds only at the BF16 stores.
The new paired-output kernel reuses the common inputs while preserving libdevice
exponential, division, derivative order, and FP32 intermediates. It is not an eager
BF16-autograd replacement, whose intermediate rounding can differ.

[One-B300 paired-gradient qualification](https://beaker.org/ex/01M1R1FT7TDYNESRG3KSX3FE3M),
source `2e9613c62`, completed exit0 at 05:45 UTC. Four launch settings passed zero-
tolerance numerical checks on empty/tail shapes, hidden sizes128/129/512/1024,
extreme values/nonfinites, and all 1,073,741,824 elements at the real gradient shape.
The best block1024/4warps backward is 0.76856ms versus ~1.77ms native before/after:
about **2.3× isolated backward speed**, ~120ms/update serial potential, not a full-model
gain. Other settings measure 0.77338/0.77912/0.79011ms. Dataset
`01M1R1FT838RC8AZ7SKA912X1E` retains the full results.

The default-off `OLMO_PROFILE_SWIGLU_PAIRWISE=1` training hook passed its
[two-B300 integration qualification](https://beaker.org/ex/01M1R1SNRYD1FP6E56NA5NZZPM)
at 05:52 UTC, source `3b020ad05`: both tests passed. The first checks compiled forward
and backward with exact output/gradient agreement and verifies that generated code
contains the new kernel. The second uses actual compiled routed experts, FP32 gradient
accumulation/reduction, and sharded Adam: three updates with eight microbatches each,
including both gradient-clearing modes. Per-microbatch losses, reduced gradients,
weights, and all local optimizer-state tensors match at zero tolerance; no skips.

That gate passed before submitting the [64-GPU activation A/B](https://beaker.org/ex/01M1R2JQX90JAQ5KGQQRHRH5QJ)
at 06:02 UTC, with [CPU collector](https://beaker.org/ex/01M1R2KKP49KZ03NS7NZ996TJG).
Source `3b020ad05`; 8 Holmes nodes, urgent/allocated1h, hosts485/516 excluded.
Fresh 60-update step7500 restores compare `kda-128-emo-inverse-scatter-grad-add`
against that same variant plus `-act-pair`, using updates31–60 and no active profiler.
All model, training, KDA, EMO, and gradient-add settings are identical across arms.
Model structure and saved activation storage remain unchanged; no additional
recomputation is introduced. The isolated 2.3× result is not yet a full-model gain.

### Nsight capture repair

The [two-B300 reproducer](https://beaker.org/ex/01M1QWYVBR97EA4G9RXCCC82TE)
completed at 04:29 UTC, source `772a914b2`. It exercises pinned H2D copies at the
actual 4×8192×1024 activation shape, forward/backward, asynchronous gradient reduction,
and model-weight gathering. Three warmup iterations precede two captured iterations.
This is a primitive workload, not the full compiled MoE.

| Profiler | Trace setup | Result |
|---|---|---|
| Installed 2025.3.1.0 | Both ranks, CUDA/NVTX/OSRT, autograd NVTX enabled | Segfault, exit139 in a traced worker |
| Installed 2025.3.1.0 | Rank0 only, CUDA/NVTX, autograd NVTX disabled | Segfault, exit139 in the traced worker |
| Standalone 2026.4.1.191 | Both ranks, CUDA/NVTX/OSRT, autograd NVTX enabled | Pass; both reports have 46 kernels, 8 memory copies, 7 NCCL kernels |
| Standalone 2026.4.1.191 | Rank0 only, CUDA/NVTX, autograd NVTX disabled | Pass; report has 46 kernels, 8 memory copies, 7 NCCL kernels |

Changing only the profiler version fixes this reproducer. Fewer ranks or disabling
autograd NVTX alone did not fix the old profiler. The exact vendor-internal defect is
not established. A `.nsys-rep` file alone is not success: the old profiler also wrote
reports after crashing. Validation now exports SQLite and requires actual CUDA kernels,
copies, and NCCL kernels. Compact result dataset: `01M1QWYVC2SDVCYSD10NP78AK5`.

The standalone installer checks NVIDIA's package SHA256 and extracts into container
`/tmp`; it does not modify drivers, CUDA, PyTorch, or host packages. The next full-model
capture uses the qualified 2026.4.1 binary, ranks0/8/16/24/32/40/48/56, and two updates
36–37 of a fresh 60-update restore. The callback and wrapper share rank/window settings;
the collector waits for the selected reports and their validation, not all 64 reports.
PyTorch and Nsight capture remain separate processes/passes.

Full-model retry: [Nsight 2026.4.1 capture](https://beaker.org/ex/01M1QXEK2Q7TEPDQZBWKAVEYKR),
source `f7c61144d`, 64 B300s, urgent/allocated1h, host485 excluded. **All eight training
tasks completed exit0 at 04:57 UTC.** Every selected rank has 68,955 CUDA kernels,
105 memory copies, 344 NCCL kernels, and 208,556 NVTX events; all eight validation
markers passed. The compiled full-model capture, not just the small reproducer, works.
[CPU-only collector](https://beaker.org/ex/01M1QXFYFH2XXTHYKHRYRBC5R5) also completed
exit0, dataset `01M1QXFYFW0KHGAKMF14D14BEQ`. Raw reports and SQLite exports stay on Weka.
This first collector also exported ~140MiB of verbose per-op NVTX statistics; subsequent
collectors retain those full statistics on Weka and publish bounded CUDA summaries.

Rank0's two-update kernel sums confirm the same large gradient-add (0.662s), expert
activation backward (0.435s), remaining EMO sort (0.220s), and FP32 router GEMMs
(0.446s) seen in PyTorch. H2D CUDA-copy sum is just 0.471ms. NCCL duration sum is
2.447s, but this is an instrumented capture, not a normal-run exposed-communication
budget. The capture/flush updates slow substantially; clean pre/post windows recover
83,938 / 83,958 median TPS/GPU on this allocation. Do not substitute Nsight's timings
for the independent same-node timing results.

The [read-only CPU timeline analysis](https://beaker.org/ex/01M1QZ3PNYDRJR54GDRBZ89RAG),
source `aac6442ea`, completed exit0 at 05:03 UTC. Dataset
`01M1QZ3PPA7ES6SGR0H8811RDP` contains eight compact JSON summaries. Rank0's two-update
kernel span is 7.536s, kernel-busy union 7.371s, collective-only union 1.923s, and
collective/other-kernel overlap 0.524s. After including recorded copies/memsets, 0.160s
has no recorded GPU operation. Across selected ranks, collective-only union ranges
1.405–1.923s and uncovered time 0.160–0.658s. These exceed the separate healthy PyTorch
capture's communication-only intervals; tracing mode and allocation differ. The data
does not justify calling this normal-run CPU starvation or predicting a removable 25%
communication cost. Before additional collective tuning, consider a reduced-overhead
Nsight repeat without per-op autograd NVTX; the repaired profiler itself is qualified.

### Next isolated candidate: optimizer model-weight gathering

`olmoe3_model_gather_bench.py` is a standalone prototype, **not wired into training**.
It compares the real optimizer packed-gather method with individual direct-to-final-view
gathers for large tensors plus bounded packed gathers for small tensors. Group routing,
FP32-master to model-dtype conversion, contiguous parameter storage, and synchronous
collectives remain unchanged. No persistent global temporary is introduced.

Six CPU/gloo layout tests passed: FP32/BF16, all-direct/mixed/all-packed cases, repeated
updates including unchanged masters, a singleton process group, replicated parameters,
unchanged master storage, and stable model/view pointers. These are gather-operation
tests, not end-to-end optimizer or training qualification.

The [two-B300 qualification](https://beaker.org/ex/01M1QVEPE4DD7TG6QGSFFPV09N)
completed exit0 at 04:02 UTC, source `c9894426f`. All six NCCL layout tests passed,
and all three benchmark arms verified every output element and unchanged source
masters. It times packed /
direct-large / packed on a **synthetic** 22.676-GiB BF16 layout containing 15 pairs of
the real 0.5/1-GiB expert tensor sizes plus synthetic smaller weights. It measures
CUDA and synchronized wall time, additional allocated-memory peaks, and verifies
every output element. The prototype makes 33 collectives instead of one, so additional
launch/network overhead may outweigh copying savings, especially at 64 ranks. A local
microbenchmark gain will not be called a whole-training gain or justify deployment.

| Isolated weight-gather arm | Median CUDA time (ms) | Median synchronized wall time (ms) | Additional peak allocated memory (GiB) |
|---|---:|---:|---:|
| Current packed path, before | 37.4641 | 37.4947 | 34.0137 |
| Direct large + coalesced small | 32.5820 | 32.6161 | 0.7500 |
| Current packed path, after | 37.4622 | 37.4938 | 34.0137 |

This is about **13.0% lower isolated gather latency**, or **4.88ms**, not 13% lower
training time. Five warmups and 20 measurements per arm, taking the slower rank for
each sample. GPU events include conversion/copy/collectives; wall time also includes
host submission and the terminal GPU wait. Input creation, barriers, and correctness
checks are outside those windows. The model/master buffers exist in every arm; the
memory column is additional live allocation, not reserved memory or model peak memory.

The two-rank local input is much larger than the 64-rank input. Accordingly neither
the latency nor the 34-GiB transient reduction transfers directly to production.
Moreover the real small model peaks during forward/backward, not this optimizer
copy stage; this result does **not** establish that MB8 would fit. The prototype
remains benchmark-only. Result dataset: `01M1QVEPEF8VKF6MC1P73HER2H`.

### Current priorities after healthy confirmation

1. Collect the faster gradient-add candidate's timing/PyTorch/light-Nsight passes.
   Confirm the new gradient-add kernel is used, measure remaining activation/GEMM
   work, and compare instrumentation overhead. Do not treat traced communication
   fractions as a normal-run removable-time budget.
2. Collect the paired-SwiGLU full-model A/B after its passed integration tests.
   Check all 60 losses/gradient norms/skips and actual allocated memory, not just TPS.
   Compare same-node updates31–60; exact primitive tests do not prove long-run quality.
3. Use the refreshed traces to choose the next grouped-GEMM or KDA optimization.
   Preserve architecture, BF16 computation, FP32 accumulation, scalable softmax,
   EMO, batch, LR, and the no-recomputation policy. Future GEMM/gradient-add fusion
   must retain BF16 weight-gradient rounding before FP32 accumulation.
4. Nsight Compute hardware counters require infra to enable permitted access
   (`ERR_NVGPUCTRPERM`). In the meantime use exact microbenchmarks and end-to-end A/Bs.
   Only pursue the packing/gather prototypes at inter-node scale if the refreshed
   trace supports it; their demonstrated isolated savings are milliseconds.

At this branch's FLOP accounting, 600 TFLOPs/GPU requires approximately 137.65k
TPS/GPU (1.904s/update), about 56.0% above the current 88.23k gradient-add result.
The small copy optimizations alone cannot close that gap. No additional 64-GPU
allocation was launched solely for either communication-copy prototype.

No experimental flag was enabled in the original CBS/production branch. No
GEMM/KDA-layer fusion, training optimizer all-gather rewrite, precision change,
recomputation, or shared EP output buffers were introduced. The paired activation
prototype is a default-off kernel rewrite, not an architecture change.

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
1.904s/update. That is about 56.0% above the latest 88,234 TPS/GPU / 384.61 TFLOPs/GPU
gradient-add result. The measured improvements so far do not close that gap.
Larger gains need measured reductions
inside each microbatch, not just an optimizer communication flag. Kernel-duration sums
must not be added across overlapping streams and treated as end-to-end time savings.

All experimental changes remain isolated in `codex/small-16mi-profile-v2`.
