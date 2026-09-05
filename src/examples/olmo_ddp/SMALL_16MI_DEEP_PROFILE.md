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

## Run ledger (2026-09-05 UTC)

- Qualification: https://beaker.org/ex/01M1QH72PZCX87XR0KA6JX5VRJ
  (one B300, source `a7848229a`). Nsight capture and source metadata checks passed.
  KDA full-layer microbenchmark uses BF16 autocast with FP32 parameter storage; it is
  a kernel qualification probe, not an end-to-end throughput claim. FLA median/mean
  10.149/10.215 ms; new default 10.084/9.952 ms. Default-path relative L2 error:
  output 0.0177%, maximum gradient error 0.169%. Lower-cutoff arm pending.
- Full profile: https://beaker.org/ex/01M1QHSZGC0V64F63Q99YVS7C4
  (`olmoe3-small-16mi-deep-profile-v2-r1`, source `3a148a86b`), queued on 64 B300s,
  urgent/allocated, one-hour minimum runtime; zero automatic task retries.
- Local checks: nine migration tests (including two-rank DCP save/load/reshard),
  fourteen attention-config/per-head-gain/scalable-softmax tests passed; lint and
  formatting checks passed for new scripts.
- Matched reference from completed 16Mi CBS history, same global step range:
  steps7531–7570 median 76,494 TPS/GPU, mean CE 1.95184; steps7581–7600 median
  76,452 TPS/GPU, mean CE 1.95538. Original step7501 CE=1.98832607,
  total grad norm=.07320063 (independent-head gains will change future gradients).

## Launch

From this clean, pushed branch, with the Beaker and Python environment configured:

```bash
python src/examples/olmo_ddp/olmoe3_small_deep_profile.py launch \
  olmoe3-small-16mi-deep-profile-v2-r1 ai2/holmes
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
