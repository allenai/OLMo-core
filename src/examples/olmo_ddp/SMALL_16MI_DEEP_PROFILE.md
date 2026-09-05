# Small production candidate: trained 16Mi profiling

This branch isolates profiling from the CBS/uploader jobs. Existing training/checkpoints are
not modified. No profile run writes checkpoints, performs evals, or registers with the uploader.

## Provenance and settings

- Base: `akshitab/moe-v2-core`, `a19a19ec32ffeb597f3ccd3ff90e623ba5ad01c7`.
- Minimal EMO/KDA/production handoff: `45c8f3f4f` (descends from that base).
- KDA PR #837 through `5fc6aa2eb`; kernel-fun pinned to
  `7a6983baf2beb4ec4d7fe914ec9f6670438af99b`. The PR's originally written full hash
  `7a6983b592cc1097b469af65953f32914cd93054` does not exist on the remote.
- Per-head QK norm gains: PR #855, `9805007358e8f5b6c131953fb270b757eedacb58`.
  Preserve core-v2 attention features when resolving its two merge conflicts.
- Small: 16 layers, d_model=1024, latent=512, Q/KV=8/4, head_dim=128,
  14 KDA / 2 FA (layers 7,15), 512 experts/top-16, expert hidden=1024.
- EMO document pool 16–512, eval pool 512, global load-balancing.
- Active 794,233,472 / total 12,496,341,632 after the 2,560 additional QK gain parameters.
- 64 B300 GPUs (8 Holmes nodes), PP1/EP1/DP64, BF16, MB4 sequences of length 8192,
  global batch 16,777,216 tokens, eight accumulation microbatches.
- FA4 with scalable softmax. No fused attention, activation checkpointing/recompute,
  MXFP8, two-batch overlap, or shared EP output buffers.
- Distributed optimizer, FP32 gradient accumulation/reduction; baseline normal-gradient
  all-reduce. `reduce-scatter` is an opt-in comparison, not the default.
- Restore full trained state from the 16Mi CBS run at step7500 (92.275B tokens), before
  final decay. WSD warmup=2000, stable LR=.00185, original 100.663296B schedule horizon.
  Stop profiling via a step hard-stop, not a shortened/decayed scheduler.
- Expand only shared Q/K gain vectors and Adam moments into per-head copies on load.
  Initial forward computation is preserved, but future independent-head optimization is
  not mathematically identical to the old shared-parameter trajectory. No source rewrite.
- Reuse the CBS data-index cache; write traces/metrics to separate run directories under
  `/weka/olmo-3p5-checkpoints/production-profiling/`, not Beaker result datasets.

## Qualification and measurements

1. CPU unit tests: strict migration whitelist, unchanged shapes, forward equivalence.
2. One GPU: runtime/dependency and checkpoint metadata checks, actual Nsight smoke capture,
   full KDA layer forward/backward parity and timing at B4/T8192/H8/K128/V256.
   Compare FLA, new default dispatch, and a diagnostic lower CTA cutoff. The lower cutoff
   is process-local in this microbenchmark; it does not alter the training default.
3. One eight-node allocation, two fresh subprocesses per rank:
   - Nsight pass: 100 updates, capture updates 71–73 on all 64 ranks.
     Unprofiled windows: 31–70 and 81–100; report both separately.
   - PyTorch pass: 60 updates, detailed trace updates 36–37 on ranks 0,8,...,56;
     memory history 45–46 on rank0. No simultaneous CUPTI consumers.
4. Analyze unprofiled steady-state TPS/step latency separately from traced event timing.
   Report mean and median, variability, peak memory, actual kernel dispatch, and losses.
   Never add overlapping kernel durations and call the sum wall time or an MFU gain.
5. Profile-driven A/B: same source checkpoint, data position, precision, model and LR;
   validate finite/matching initial loss and gradients before claiming a speed improvement.

At MB4 the new KDA package's default 256-CTA cutoff rejects our 128-CTA scan grid and
falls back to FLA. Short convolution can still dispatch to the new kernels. Verify logs;
do not describe the whole model as running the new CuTe KDA chain without evidence.

## Initial hypotheses, not measured findings

Earlier matched-token CBS measurements: ~76.4k TPS/GPU at 16Mi, ~3.43s/update.
A simple fit across 4/8/16/32Mi gives ~0.34s fixed update cost plus ~0.387s/microbatch;
eliminating only that fitted fixed cost offers about 11% throughput improvement.
600 TFLOPs/GPU would require roughly 138k TPS/GPU (~1.9s/update), so microbatch work
must improve substantially too. Use the same model FLOP accounting as earlier runs.

Investigate: expert GEMMs and EMO packing/permutation; KDA and short convolution;
FP32 gradient accumulation copies; CPU launch gaps; exposed gradient reductions and
optimizer parameter all-gathers. EP1 has no expert-dispatch all-to-all, but still has
data-parallel communication over the full 12.5B-parameter model.

Candidate interventions, only as supported by measurements: normal-gradient reduce-scatter,
bucket sizing/communication scheduling, compiled/fused pointwise work compatible with
per-head gains and scalable softmax, KDA dispatch tuning. Keep precision/model changes
and recomputation out of this baseline.

For final small-gain claims, run plain timing arms on the same allocation, with no CUPTI
profiler in any arm: `OLMOE3_DEEP_PROFILE_PASSES=timing` and
`OLMOE3_DEEP_PROFILE_VARIANTS=baseline,reduce-scatter,kda-128`. Each arm restores the same
checkpoint/data position and runs 60 updates; compare updates31–60. Each variant/pass gets
a separate torchrun agent and rendezvous port. This also avoids reusing store keys across
fresh training processes. The reduce-scatter arm changes communication, not precision or
model/optimizer math; measure its packing overhead and numerical agreement too.

Additional compiler candidate: actual compilation logs show `_NoOpRange` contexts preventing
Dynamo from resuming around intentional graph breaks in the MoE path. A minimal reproducer
with an opaque call inside the disabled annotation produces zero compiled regions; using
`nullcontext` permits two compiled regions around the same opaque call, with identical
outputs. The opt-in `compile-noop-nvtx` variant enables `OLMO_PROFILE_SAFE_NOOP_NVTX=1`:
disabled decorators become identity decorators, and compiled disabled contexts use the
standard nullcontext. Real NVTX annotations are unchanged. Two local tests cover identity,
exception propagation, compiler resumption, and exact output agreement. End-to-end speed
and numerical validation remain required; the flag is off in the baseline.

## Run ledger (2026-09-05 UTC)

- Qualification: https://beaker.org/ex/01M1QH72PZCX87XR0KA6JX5VRJ
  (one B300, source `a7848229a`). Nsight capture and source metadata checks passed.
  KDA full-layer microbenchmark uses BF16 autocast with FP32 parameter storage; it is
  a kernel qualification probe, not an end-to-end throughput claim. FLA median/mean
  10.149/10.215 ms; new default 10.084/9.952 ms. Default-path relative L2 error:
  output 0.0177%, maximum gradient error 0.169%. Lower-cutoff arm passed:
  median/mean 8.992/8.986 ms, output error 0.404%, maximum gradient error 0.570%.
  This is ~10.8% less layer time than new default, not a whole-model gain.
- Full profile: https://beaker.org/ex/01M1QHSZGC0V64F63Q99YVS7C4
  (`olmoe3-small-16mi-deep-profile-v2-r1`, source `3a148a86b`), canceled before training.
  Beaker replaced rank0 after a health-check failure (host516 -> host526), but the other
  seven jobs retained the old injected leader hostname and stalled in Gantry rendezvous.
- KDA cutoff comparison: https://beaker.org/ex/01M1QJ5SRMTM4GFMDRWSA0Z00Y
  (`olmoe3-small-16mi-kda128-v2-r1`, source `b2892e0f0`), 64 B300s, 60 updates,
  timing only; canceled while queued so it can use the same corrected launcher.
  Same model/checkpoint/batch/LR/precision, with cutoff128 enabled only here.
- The corrected node launcher queries current Beaker assignments after setup and waits
  for all eight current job-ID-specific readiness markers before invoking torchrun.
  It retains host networking, leader selection and failure propagation; stale injected
  hostnames and markers from replaced jobs are not used for rendezvous.
- Replacement full profile: https://beaker.org/ex/01M1QJT2NVG2WXVNEPQ1KPDT5M
  (`olmoe3-small-16mi-deep-profile-v2-r2`, source `e90ee93ee`).
- Replacement cutoff comparison: https://beaker.org/ex/01M1QJT0K2X5P2JCCWXVQS8GP2
  (`olmoe3-small-16mi-kda128-v2-r2`, same source). Both urgent, allocated, 64 B300s.
- Both r2 attempts passed rendezvous but stopped before any training updates: the
  harness inherited `no_checkpoints=True`, which skips loading as well as saving.
  The expected-step guard rejected step0. Corrected to enable checkpoint loading with
  an explicitly disabled `CheckpointerCallback`, so no new checkpoints are written.
- CPU collector r2: https://beaker.org/ex/01M1QJZ5TH49B512H9366H5753; canceled when its
  upstream r2 jobs failed. No GPU resources used by the collector.
- Corrected full profile r3: https://beaker.org/ex/01M1QKFYH4HKM3BEK7TJ2A31YN
  (`olmoe3-small-16mi-deep-profile-v2-r3`, source `b0124be9c`). Beaker replaced replica5
  after an interconnect health-check failure, before any model code ran.
- Full trained-state restore succeeded at 01:51 UTC, including the twelve gain/moment
  expansions. Dry-run compilation took ~7.7 minutes; later dry-run microbatches were
  ~0.38s each. Early real updates settle near 76.5k TPS/GPU / 333 TFLOPs/GPU, with
  finite losses and zero skipped steps. These are provisional logs, not the final
  clean-window statistics or a performance improvement claim.
- Same-allocation timing matrix: https://beaker.org/ex/01M1QM9ZVJG0CB5B2609FW77SX
  (`olmoe3-small-16mi-timing-matrix-v2-r3`, source `41084cc46`), baseline then
  reduce-scatter then cutoff128, 60 updates each from the same checkpoint.
- Its baseline completed all 60 updates. Clean updates31–60: median/mean
  78,222/78,081 TPS/GPU, median 340.965 TFLOPs/GPU, median step 3.35128s.
  Mean CE=1.951015, zero skipped steps; mean data loading=.002708s/update.
  Per-step rank-reduced memory: 175.466 GiB active, 178.563 GiB reserved.
  End-of-run allocator counters are **not** full-run peaks: the standard memory
  callback resets them each step. Use logged per-step peaks or memory snapshots.
- Nsight r3 failed immediately after capture began at step7571: rank55 segfaulted
  in the CUDA host-to-device copy path, and sibling jobs then stopped. Updates31–69
  before capture remain usable timing data (39 updates, median 76,729 TPS/GPU),
  but this is not a completed 100-update run. The surviving report has no CUDA data;
  no kernel-level conclusions can be drawn from it.
- Independent PyTorch capture replacement:
  https://beaker.org/ex/01M1QNHVHXXPSJHBFF7KBB7SCV
  (`olmoe3-small-16mi-torch-profile-v2-r4`, source `dbaa6d5f9`).
  Completed: all eight jobs exited0; all eight representative-rank Chrome traces,
  distributed-event summaries, and rank0 memory snapshot exist (about 176 MiB total).
  Interpretation and the profile-vs-unprofiled timing caveat are recorded in
  [SMALL_16MI_PROFILE_RESULTS.md](SMALL_16MI_PROFILE_RESULTS.md).
- Same-node compiler-no-op A/B:
  https://beaker.org/ex/01M1QNHSC6ZVAGBGV38KAGW605
  (`olmoe3-small-16mi-noop-ab-v2-r1`, source `dbaa6d5f9`), baseline then
  `compile-noop-nvtx`, 60 unprofiled updates each, both completed. Median TPS/GPU
  76,921 -> 77,276 (+0.46%), zero skips. This small single-pair difference does not
  establish a substantial gain. Experimental flag remains off by default.
- Partial recovery completed successfully:
  https://beaker.org/ex/01M1QNM5FA1E56B4C50T6P1XJE
  (result dataset `01M1QNM5FGV2W0BVVNVHEHE068`). A redundant replacement was stopped.
- Active CPU collector:
  https://beaker.org/ex/01M1QNWGMB6NHEQPSCPKP45C16, collecting all six timing/torch
  passes independently as they complete. It skips package installation because its
  analysis uses only the standard library and the existing Nsight binary. Previous
  collector r5 was stopped/replaced; its first timing summary is retained in dataset
  `01M1QNM636RNYZPXQ33V9NPMTY`.
- The dry run consumes EMO RNG draws. All comparison arms use the same resume and
  warmup procedure, but this is not a bitwise replay of uninterrupted CBS training.
- CPU collector `01M1QMAARSW1HQ6EPFM5NJ46C1` was stopped before upstream completion
  to replace it with a version that exposes intermediate summaries directly in logs;
  training allocations were not changed.
- Local checks: nine migration tests (including two-rank DCP save/load/reshard),
  fourteen attention-config/per-head-gain/scalable-softmax tests passed; lint and
  formatting checks passed for new scripts.
- Matched reference from completed 16Mi CBS history, same global step range:
  steps7531–7570 median 76,494 TPS/GPU, mean CE 1.95184; steps7581–7600 median
  76,452 TPS/GPU, mean CE 1.95538. Original step7501 CE=1.98832607,
  total grad norm=.07320063 (independent-head gains will change future gradients).
- EMO inverse-scatter GPU qualification r2 passed:
  https://beaker.org/ex/01M1QQ1MGM97EYNDDTZ379CPWE (source `c0f49384f`).
  The first attempt's eager-vs-compiled tie comparison was invalid even for the
  unchanged baseline; corrected comparisons require exact equality within each mode.
  Compiled median mask time 1.7313ms -> 1.1354ms, not an end-to-end speedup claim.
- Same-node EMO full-model matrix completed (all eight jobs exit0):
  https://beaker.org/ex/01M1QQCYKN4AGBMRHX84618N6C (source `680ed4ae2`).
  Independent baseline, `emo-inverse-scatter`, `kda-128-emo-inverse-scatter` arms,
  each 60 unprofiled updates from step7500. Analysis automation:
  https://beaker.org/ex/01M1QQEP0RE71VBZZA2CRHXJ6V (CPU only, unallocated).
  The latter also completed the enhanced PyTorch trace analysis, and all results were
  downloaded in dataset `01M1QQEP128HQJHQXDJEBH1FFB`. Median TPS/GPU was 58,832 baseline,
  60,991 EMO (+3.67%), 63,007 combined (+7.10%), zero skips. This allocation and the
  earlier capture included host485's disconnected NVLink GPU; comparisons must be
  confirmed on healthy nodes before claiming production throughput.
- Healthy confirmation/capture queued at source `5c7d46b4d`:
  https://beaker.org/ex/01M1QT33FG4X90DBC4WE44ZP3V. Three unprofiled arms (baseline,
  KDA128, KDA128+EMO) then one combined PyTorch capture, all on the same eight nodes.
  Host485 excluded and each node must pass the full NVLink topology check.
  CPU result collector: https://beaker.org/ex/01M1QT82PQS02QRDBM3KDSF63X.
  All eight training jobs had started by 03:44 UTC; no completed timing yet.
- Two-GPU RS qualification: https://beaker.org/ex/01M1QT7ZYNN58QK72YMY2CKDXK,
  source `92a24ffa0`, completed exit0. Nine new NCCL parity cases and three existing
  gradient/optimizer tests passed (two four-GPU tests skipped). Single-parameter
  bucket operation medians: 1GiB 2.157->1.461ms, 2GiB 4.158->2.823ms; repeated old
  arms 2.151/4.149ms. Approximately 32% lower isolated latency, not training TPS.
  Result dataset `01M1QT7ZYYQY7EXGNFZAG642PM`. First attempt
  `01M1QT0W0CRGTESJNZ7M0KTNHT` stopped at a full-eight-GPU guard in a two-visible-GPU
  container. The corrected explicit `--gpus 2` option does not weaken full-node checks.
- PR859 reviewed at `fe0195dd845a3dd786e4fc80d52eed9e7ce404b0`:
  its 23 vendored kernel files are byte-identical to our installed `kernel-fun`
  revision `7a6983baf2beb4ec4d7fe914ec9f6670438af99b`. It is an alternative
  distribution, not another kernel implementation to benchmark. No duplicate merge
  or GPU job. See the results document for the exact comparison and implications.

## Launch

Before new comparisons, populate `OLMOE3_ALLOWED_HOSTNAMES` with eligible Holmes hosts
**excluding `holmes-cs-aus-485.reviz.ai2.in`** pending NVLink repair. On 2026-09-05 its
GPU1 was disconnected from NVLink and caused an eight-node job to use nine NCCL nodes,
four collective channels and no NVLS. The node driver now rejects any local eight-GPU
topology containing off-diagonal non-NVLink paths before workers compile.

Standalone follow-up on 2026-09-05 (source `c9894426f`):
[optimizer model-gather qualification](https://beaker.org/ex/01M1QVEPE4DD7TG6QGSFFPV09N),
two B300s / one Holmes node, urgent allocated10m in `ai2/olmo3p5-training`, no data
or checkpoint mounts. Completed exit0 at 04:02 UTC; six NCCL layout tests passed.
Synthetic 22.676-GiB packed/direct/packed gather medians: 37.464 / 32.582 / 37.462ms.
This prototype lives only in the benchmark, not optimizer/training configuration.
Result dataset `01M1QVEPEF8VKF6MC1P73HER2H`; full caveats in the results report.

`OLMOE3_DEEP_PROFILE_PLAN` optionally specifies an ordered `variant:pass` list instead
of the Cartesian product of variants and passes. For example:

```text
baseline:timing,kda-128:timing,kda-128-emo-inverse-scatter:timing,kda-128-emo-inverse-scatter:torch
```

Each pass is a fresh trained-step7500 restore on the same eight nodes, with unique
artifact paths. This allows controlled unprofiled comparisons followed by a short
capture without profiling every candidate. Duplicate pairs are rejected.

From this clean, pushed branch, with the Beaker and Python environment configured:

```bash
python src/examples/olmo_ddp/olmoe3_small_deep_profile.py launch \
  olmoe3-small-16mi-deep-profile-v2-r2 ai2/holmes
```

The launcher uses `ai2/olmo3p5-training`, urgent priority, allocated with a one-hour
minimum runtime. The two profiling passes share the allocation and do not run concurrently.

One-GPU qualification uses the same launcher with overrides:

```bash
python src/examples/olmo_ddp/olmoe3_small_deep_profile.py launch \
  olmoe3-small-profile-qualify-v2-r1 ai2/holmes \
  --launch.num_nodes=1 --launch.num_gpus=1 --launch.torchrun=false \
  --launch.min_runtime=15m \
  '--launch.cmd=[python,src/examples/olmo_ddp/olmoe3_profile_preflight.py]'
```

For a later timing-only reduce-scatter comparison, set
`OLMOE3_DEEP_PROFILE_PASSES=timing OLMOE3_DEEP_PROFILE_VARIANT=reduce-scatter` and use
a new run name. Do not compare a traced step against an untraced baseline step.

The corresponding controlled KDA heuristic comparison uses
`OLMOE3_DEEP_PROFILE_PASSES=timing OLMOE3_DEEP_PROFILE_VARIANT=kda-128`.
This lowers only the pinned package's performance cutoff, not the model dimensions,
batch size, precision, LR, or checkpoint. The completed 60-update comparison improved
median TPS by 5.03%, with finite losses and no skipped steps; long-run stability is
not yet established. It is not enabled in the baseline.

`olmoe3_profile_collect.py` can run in a separate CPU-only job with the checkpoint Weka
mount. It waits for completed passes, analyzes traces there, and copies only small JSON/
CSV-like summaries to Beaker results. Raw Nsight/Chrome/memory traces remain on Weka.

## Nsight repair and next sequence (2026-09-05 06:03 UTC)

Two-GPU repro `01M1QWYVBR97EA4G9RXCCC82TE` (source `772a914b2`) reproduced the installed
2025.3.1 segfault in both matched/full and reduced tracing settings. The identical
workload passed both settings under standalone 2026.4.1.191, with real CUDA/copy/NCCL
events validated from SQLite. No driver/runtime or model workaround is needed for this
reproducer. Full compiled-MoE verification `01M1QXEK2Q7TEPDQZBWKAVEYKR` (source
`f7c61144d`) completed exit0 on all eight nodes at 04:57 UTC. All eight selected
reports have 68,955 kernels, 105 memory copies, 344 NCCL kernels, and valid SQLite
exports. CPU collection also completed, dataset `01M1QXFYFW0KHGAKMF14D14BEQ`.

The current healthy same-node unprofiled result is 78,115 → 82,190 → 85,888 median
TPS/GPU for baseline → KDA cutoff128 → KDA+EMO inverse scatter, respectively.
All use step7500, 60 updates, and the same BF16 model/training settings. The combined
candidate is +9.95% over baseline, 374.38 idealized TFLOPs/GPU. Its separate healthy
PyTorch capture completed at 04:33 UTC; use that instead of the earlier degraded-fabric
capture to prioritize work. Nsight has substantial capture/flush overhead, so preserve
the clean timing windows and do not use its raw communication fractions as production
critical-path estimates.

Next actions:

1. Vectorized FP32 gradient addition completed the full-model A/B
   `01M1QYS6RMFNV0BP28QXW16W4A`, source `1c3d41fac`, with collector
   `01M1QYT2B1XGS0QPMQ1R901X9N`. Two-GPU sharded-Adam qualification passed exact
   gradients/weights/states plus four existing NCCL tests. The isolated kernel is
   2.12–2.13× faster; serial extrapolation is ~158ms/update, not 2× training. Both A/B
   arms use the same trained restore/allocation and timing window. Check all 60 losses,
   skipped steps, first-update agreement, memory, and gain versus timing variability.
   **Result:** 84,523 → 88,234 median TPS/GPU (+4.39%), 130.44ms/update saved,
   384.61 idealized TFLOPs/GPU. All 60 updates per arm finite, no skipped steps, same
   active memory. Full trajectories are not bitwise identical; exact primitive/DDP
   checks do not establish long-run quality. Dataset `01M1QYT2B7DDJF9AR8HYY70CDH`.
   Continue with fresh unprofiled timing and separate torch/light-Nsight captures of
   the faster candidate; require its expected Triton gradient-add kernels in the trace.
   These are running as `01M1R17MJ5BNM7QS6NPVTBJQY8`, source `a401daccf`, with CPU
   collector `01M1R17XVBFB9MNDG4MRX93Y28`. Nsight 2026.4.1 captures ranks0/8/.../56
   at updates36–37 with per-op autograd NVTX disabled; application NVTX remains.
2. Compare healthy PyTorch and repaired Nsight timelines for overlap and host waits,
   keeping instrumented versus clean timing separate. CPU-only SQLite analysis
   `01M1QZ3PNYDRJR54GDRBZ89RAG` completed exit0, dataset `01M1QZ3PPA7ES6SGR0H8811RDP`.
   It merges intervals per device; it never adds overlapping kernels into a step-time budget.
   Nsight's exposed-collective intervals exceed the separate PyTorch capture; a lighter
   follow-up without per-op autograd NVTX should precede conclusions about communication.
   Prioritize expert GEMMs and activation/backward traffic after the gradient-add result.
   The one-B300 activation probe `01M1R18061X37H5332NRN9EMCZ` completed exact
   outputs/gradients under default / pointwise-coordinate tuning / default, with
   generated-code export. Forward+backward: 2.26384 / 2.16994 / 2.26381ms. No global
   compiler flag changed. Generated backward repeats loads/exponentials across the
   concatenated gradient halves and keeps intermediates FP32 until the BF16 stores.
   A paired-output implementation preserves this generated arithmetic and passed
   raw one-GPU checks `01M1R1FT7TDYNESRG3KSX3FE3M`, including all 1,073,741,824 real-
   shape output elements and empty/tail/nonfinite cases. Backward 1.77 → 0.769ms,
   ~2.3× isolated speed, not a training gain. Source `2e9613c62`.
   Default-off integration at `3b020ad05` passed both GPU tests in
   `01M1R1SNRYD1FP6E56NA5NZZPM`: compiled forward/backward plus actual compiled routed
   experts with FP32 accumulation/reduction and sharded Adam, three eight-microbatch
   updates, exact losses/gradients/weights/states. The full-model timing A/B is now
   submitted as `01M1R2JQX90JAQ5KGQQRHRH5QJ`, collector `01M1R2KKP49KZ03NS7NZ996TJG`.
   Both arms keep KDA128/EMO-inverse-scatter/vectorized-grad-add and change only
   `OLMO_PROFILE_SWIGLU_PAIRWISE`, independently restoring step7500 for 60 updates.
   FP32 router GEMMs are visible but changing their precision is not a free optimization.
   Any future GEMM/gradient-accumulator fusion must preserve the existing intermediate
   BF16 gradient rounding before FP32 addition; direct unrounded FP32 accumulation is
   a numerically different experiment, not an exact replacement.
3. Hardware-counter probe `01M1QXWSA88MHCEG947BFK6VZS` established that Nsight Compute
   is blocked by `ERR_NVGPUCTRPERM`, not a crashing workload. Ask infra to enable permitted
   GPU counter access; do not bypass host controls. Once available, counter-profile
   isolated hot kernels with representative shapes/routing, not replay of NCCL training.
   In the meantime, exact-parity microbenchmarks and full-model timing A/Bs can continue.
4. If the healthy timeline supports it, qualify the already-tested communication-copy
   prototypes at larger rank count and in an optimizer step. The two-GPU bucket/gather
   improvements are milliseconds, not 13–32% end-to-end gains. Preserve asynchronous
   buffer lifetimes and parameter layout; defaults remain unchanged.

Future launch allowlists exclude hosts485 and516. Host485 has the diagnosed disconnected
NVLink GPU. Host516 separately failed Beaker's pre-start health check twice and was
replaced before training; the same underlying hardware fault is not established.
No administrator-level host changes were made. Every full-model launch still requires
the complete eight-GPU local NVLink preflight on each assigned node.

Selective Nsight launch controls (in addition to the normal launch command and healthy
host allowlist):

```bash
export OLMOE3_DEEP_PROFILE_PASSES=nsys
export OLMOE3_DEEP_PROFILE_VARIANT=kda-128-emo-inverse-scatter
export OLMOE3_DEEP_PROFILE_STEPS=60
export OLMOE3_NSYS_VERSION=2026.4.1
export OLMOE3_NSYS_RANKS=0,8,16,24,32,40,48,56
export OLMOE3_NSYS_START=36
export OLMOE3_NSYS_END=37
export OLMOE3_NSYS_TRACE=cuda,nvtx,osrt
export OLMOE3_NSYS_AUTOGRAD_NVTX=0  # lighter follow-up; historical/default value is 1
```

The package is downloaded once per selected node and SHA256-verified/extracted in
private container `/tmp` before rendezvous readiness. Collector uses the same version
for report statistics. Raw reports and SQLite exports remain on Weka.
