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
  `compile-noop-nvtx`, 60 unprofiled updates each. Experimental flag remains off
  by default and has not yet established a speedup.
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
- Same-node EMO full-model matrix launched:
  https://beaker.org/ex/01M1QQCYKN4AGBMRHX84618N6C (source `680ed4ae2`).
  Independent baseline, `emo-inverse-scatter`, `kda-128-emo-inverse-scatter` arms,
  each 60 unprofiled updates from step7500. Analysis automation:
  https://beaker.org/ex/01M1QQEP0RE71VBZZA2CRHXJ6V (CPU only, unallocated).
  The latter also completed the enhanced PyTorch trace analysis.
- PR859 reviewed at `fe0195dd845a3dd786e4fc80d52eed9e7ce404b0`:
  its 23 vendored kernel files are byte-identical to our installed `kernel-fun`
  revision `7a6983baf2beb4ec4d7fe914ec9f6670438af99b`. It is an alternative
  distribution, not another kernel implementation to benchmark. No duplicate merge
  or GPU job. See the results document for the exact comparison and implications.

## Launch

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
