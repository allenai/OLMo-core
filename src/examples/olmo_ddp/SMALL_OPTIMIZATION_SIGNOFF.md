# Small-model optimization and integration sign-off

Authorized 2026-09-05 07:12 UTC. Stay within this experiment set. Delete nothing
on Weka, including through training retention or uploader auto-delete. No
activation checkpointing, architecture/precision change, or async checkpoint work.
All launches come from clean committed/pushed source. No production/CBS changes.

## Completion condition

Explore the six trace-supported targets below, keep worthwhile qualified changes,
then launch reference and combined-candidate production-setting integration runs:
64 B300 GPUs each,16,777,216 tokens/update,MB4,PP1/EP1/DP64,6,000 updates
(100.663296B tokens),LR0.00185,WSD with2,000 warmup steps. Identical architecture,
initial weights/data order/eval and synchronous-checkpoint schedules, uploader
enabled with deletion disabled. Confirm actual training starts, or leave queued
after a successful relevant smoke test if allocation is unavailable. Do not stop
merely because one exploratory idea fails or finishes. One consolidated KDA PR is
authorized for reusable improvements; no automatic merge.

## Baselines and gates

- Per-feature performance reference: paired-activation candidate (~91.8k TPS/GPU).
- Final baseline: same fixed model/EMO/QK-per-head/package versions and correctness
  fixes, but none of this profiling effort's performance changes (CTA128,
  inverse-scatter, vectorized gradient-add, paired activation, or later features).
- First A/A: two independent200-update restores each of pre-paired reference and
  paired candidate, interleaved on the same64-GPU allocation, source step7500.
- Exact primitive and compiled routed-expert/optimizer tests precede full-model
  trials. A/A defines observed full-run variability, not permission to ignore a
  reproducible numerical defect. Combined short integration gate before100B jobs.

## Work queue

1. EMO pool selection once per document instead of repeated token-wise sorting.
2. Specialized512-expert/top16 routing, preserving masks, tie and RNG semantics.
3. Targeted weight-gradient GEMM/FP32 accumulation fusion, preserving BF16 rounding.
4. KDA backward intra/WY/intermediate-traffic investigation; no blind failed tiling
   repeat, unsupported chunk-size or precision changes.
5. GEMM–activation fusion with the original BF16 rounding boundaries retained.
6. Communication overlap and packing based on measured exposed work. No blind
   protocol overrides or shared EP buffers. Ordinary reduce-scatter already tied.

Use independent1–2 GPU correctness/latency probes concurrently where possible.
Only promising candidates get64-GPU timing. Record rejected ideas as well as wins.
Trace-supported additional ideas are in scope; unrelated jobs are not.

## Live ledger

- Starting source:2b57788dc, clean branch `codex/small-16mi-profile-v2`.
- A/A200 four-arm control: `01M1R6VP5BYAKDABEE8B5Q03JT`, source `c51e30792`,
  64 GPUs urgent/allocated; collector `01M1R6TW08XDS8EGXY8CBTVKBB` (0GPU).
- Document-pool one-GPU compiled exact-mask/latency probe prepared, benchmark-only.
- Document-pool probe `01M1R70BRD9F5E842SPPB005AF`, source0317e7ec1, exit0:
  all10 random/tied cases exact. At16/128/1024 tokens/document, about1.1ms→0.15–0.16ms;
  length1 regresses1.15→1.42ms. Qualify mixed lengths and actual router before full-model.
- Rounded BF16 weight-gradient/FP32 accumulation epilogue probe prepared; benchmark-only.
- Rounded probe `01M1R79CDKFTF1DP9W26ASP52H`, source8b1d2d52a, exit0:
  eight exact accumulation updates, four uniform/skew up/down cases. Isolated
  GEMM+accumulation latency reduced17–34%; not yet a training gain.
- Document router r1 `01M1R7GYA4SBK4SJ19P54X45WC`: exact-only compiled check
  failed on FP32 weights (first max difference5.96e-8). Diagnostic A/A + candidate
  `01M1R88TFFBH9N28DMYDHZGT3X`, sourcec4b64e137, passed: zero routing-index/count
  mismatches through3x8 microbatches/Adam updates; A/A exact. Candidate max relative
  L2 input-gradient4.66e-5, router gradient1.20e-6, parameter1.96e-6. Numerically
  close, NOT bit-identical. Local results: profiling-analysis-20260905/document-router-r2.
- Document-pool full-model A/B `01M1R8KM6BCVZCWGB09QS96QG2`, sourcea97e8b133,
  64GPU urgent/allocated,60updates per arm; collector `01M1R8RRV3921XMJ8F6WPA1GVW`.
- Top16 first probe `01M1R8EQA6EQ4E2RSVCC60HQCH`: ~0.71ms→0.18ms, but tie
  ordering differed (even4 random-case indices). Rejected as-is. Native-tie
  emulation r2 had a Triton constexpr-reassignment compile error; corrected r3
  `01M1R94AYDN2PK92SX2G4M2RYC`, source69bd7c392, pending.
- KDA WY r1 lacked pinned FLA; r2 corrected FLA0.5.2 but rejected4-warps schedule
  on dbeta differences. r3 `01M1R8RGFPSN1T81T83ZRVE4RH`: main stages2/4 exact for
  weak/strong decay with beta>1, but host-driven whole-chain timings did not show
  a reproducible improvement. Device-time graph diagnostic r4
  `01M1R94GAXA9FP1M1VZS00TXS2`, source69bd7c392. No schedule promoted yet.
- Rounded-wgrad DDP prototype source0f68f5634: dedicated QuACK compiler/cache,
  explicit bucket ownership/completion (no fake gradients), fail-closed reuse and
  mixed-native-gradient guards.2GPU qualification `01M1R945X0FWPBZH0SE540T5EE`,
  source69bd7c392. Local CPU regression9passed4GPUskipped. Not full-model enabled.

## September 5, 08:48 UTC results and follow-through

- A/A200 completed all four arms successfully. Median TPS/GPU: reference repeat1
  87,209, repeat2 87,627; paired repeat1 90,325, repeat2 90,656 (~3.5% gain).
  Mean absolute per-step CE A/A difference: reference .001210, paired .001073;
  paired-v-reference A/B .001115/.001080. Similar observed loss variability,
  not proof of exactness. Paired repeat2 skipped step7645 (norm .203071); other
  arms had no skips. Preserve this outlier for long-run sign-off, do not hide it.
- Document-pool A/B completed: 92,236 ->95,376 median TPS/GPU (+3.4%),
  402.05 ->415.74 TFLOPs/GPU; mean absolute CE difference .0003983, max .002128,
  no skipped updates. All60 steps finite. Local `profiling-analysis-20260905/docpool-ab-r1`.
- Top16 native primitive r3 passed exact indices/values at ~.259ms vs .71ms.
  Combined router initial synthetic test failed two reordered entries after
  independent Adam trajectories. Original synthetic loss attached signal to
  top-k slot rather than expert identity, creating artificial reorder sensitivity.
  Fixed test separates same-weight correctness from independent trajectories and
  gathers expert-identity loss signals. `01M1RBNHSQB7YHMT170TVA2CMB`, source34757f9df,
  passed4 tests (all24 primitive cases, ragged docs, same-weight and trajectory
  router/gradient/Adam gates). Same-weight requires exact order; trajectory permits
  order changes but still requires the same selected sets/counts and numeric bounds.
- Rounded-wgrad two-GPU qualification `01M1R945X0FWPBZH0SE540T5EE` passed3 tests:
  exact losses, FP32 accumulation/reduction, weights and Adam states through3x8.
  Full A/B r1 `01M1R9QBA0A02NCE5YRXQ4BPVH` reference ran, candidate failed before
  training because its WandB tag was65 chars. Fixed bounded tags, no kernel fix.
  Full A/B r2 `01M1RBP5RSDYB622KG4PERXGMJ`, source34757f9df, launched.
- RS packing requalification `01M1RBNQGDSW0Y630E1Z79JV0G`, source34757f9df:
  correctness suites passed, latency benchmark still running at08:48.
- KDA WY r4 full graph capture hit the package's intentional graph fallback.
  r5 `01M1RA42V2T8FXN4XXEPHNYASG` captures only pure-Triton WY using real inputs:
  baseline3stages ~.597-.600ms;2stages ~.613-.615ms;4stages ~.709-.712ms.
  Before/after baseline matched. Reject2/4stages for speed; reject4sidewarps for
  dbeta differences. No new WY schedule promoted.
- GEMM/SwiGLU r1 `01M1RA3W3NXQDZKPE6Q770X1NY` hit QuACK concat-layout + ragged
  saved-preactivation TMA layout incompatibility, before numerical/timing checks.
  Do not enable unrounded/default QuACK SwiGLU as a substitute.
- Integration configs prepared, not yet launched: same fresh seeds12536/928543231,
  same serialized model/training/data configs verified locally; private shared
  uploader bucket, unique run prefixes and mandatory report_only registration.
  50TB volume has~40.25TB free; two25-checkpoint runs require~7.50TB, plus smokes.
  Smoke plan: reference0->4->8, optimized0->4->8, identical allocation, sync
  step0/4/8 saves, short held-out eval and first-batch/initial-weight fingerprints.

## September 5, 09:40 UTC: combined qualification

| Isolated full-model A/B | Reference TPS/GPU | Candidate TPS/GPU | Gain |
|---|---:|---:|---:|
| Document pool |92,236|95,376|3.4%|
| Rounded wgrad, r2 |91,907|95,547|4.0%|
| Native-tie top16, atop docpool |92,951|94,352|1.5%|
| Direct single-parameter RS packing |90,460|93,582|3.5%|

Each row uses its own same-allocation60-update reference, median steps31–60;
do not multiply these gains. All60 updates finite, none skipped. Experiment IDs:
wgrad `01M1RBP5RSDYB622KG4PERXGMJ`, top16 `01M1RC507P0RQ99VX03X3NJPEQ`,
RS `01M1RC5N6V4FWPA6PWHV8KQS2H`. Wgrad CE mean/max absolute delta
.000258/.001629; top16 .000824/.007127; RS .000516/.002697. The top16 maximum
slightly exceeds the prior reference A/A maximum .006603; it is not correct to
say every single-step difference is within that envelope. Combined repeats follow.

- Combined rounded-wgrad+RS ownership/optimizer test `01M1RE5SV6XQFGMMJT5KSQD1F8`,
  source35fbbc3a9, passed all3 two-GPU modes: AR, packed RS, direct RS. Exact
  losses, full gradients, reduced shards, parameters and Adam state through3x8.
- Combined200-update A/A `01M1RERW7CSCBHWKDDGEED3AFT`, source35fbbc3a9:
  original baseline, combined candidate AR, combined candidate direct RS, each
  repeated twice on one64-GPU allocation. Candidate includes CTA128, inverse
  scatter, vectorized FP32 add, paired activation, docpool, top16 and rounded wgrad.
- Paired fresh save/restore/eval smoke `01M1RESCKKEN94W5NK0RC9F5Z2`, same source:
  reference and candidate-RS each0->4->8, separate torchrun agents,64 GPUs.
  CPU collector compares64 initial-weight and128 batch fingerprints per arm,
  six completed checkpoints and six remote-verified uploads. No deletion.
- Four report_only integration registrations created by
  `01M1RC7QKW6RMX36A1CN6BSA56`; live uploader discovered them without restart.
- GEMM/SwiGLU fusion r2 `01M1RDF824AB3XZK1XC14FB8MP` corrected saved-output
  N-layout compilation but failed with CUDA illegal instruction before validation.
  Reject as-is; no such code is enabled in the training candidates.
- Consolidated KDA contribution opened: https://github.com/allenai/kernel-fun/pull/2
  (private fork branch `codex/kda-cta-policy-b300`, ec10bc7). Opt-in top-level
  CTA dispatch policy; default256 and all safety/internal-kernel guards retained.
  Twelve CPU tests and two exact B300 forward/backward production-shape tests
  passed (`01M1RCTRHBHVSGXRFQRYVFPTRZ`). No WY schedule win promoted, no merge.
  Training still pins7a6983 and the already-qualified CTA128 override.

Smoke first attempt failed before training/checkpoint creation: validation was
incorrectly addressed as `s3://ai2-llm`, yielding403 on metadata HEAD. The correct
location is `gs://ai2-llm`. All11 NPY files and11 sidecars verified with metadata
and byte-range reads using the existing workspace Google credential (not printed).
Only the validation URI changes; train data remains Dolma3.5 S3. Retry uses the
same untouched smoke roots/registrations. Audit collector r2
`01M1RF8RSF9GN7T976PB7EDFPF` handles explicitly unvisited partial-eval subsets and
requires the current manifest's digest to match the read-back-verified publication.

- Corrected smoke retry: `01M1RFHFA1GFSQMDN4Q3QRQ15B`, source811cdf62a.
- Extra trace-supported router FP32 BLAS probe: `01M1RFWPEHENKEQXW1MV7ZWN0M`,
  sourced88078408, one B300, exit0. cuBLAS / cuBLASLt / cuBLAS bracketing at
  M32768,D1024,E512, all precision=highest and TF32 disabled. All three dispatches
  used the same actual CUDA kernels and produced bit-identical FP32/BF16 results.
  Forward~.613ms, input-gradient~.558ms, weight-gradient~.646ms. No useful gain;
  no global BLAS preference change in training. Artifacts:
  `profiling-analysis-20260905/router-blas-r1`, dataset01M1RFWPESG0YGM3RW92AVRY99.

## September 5, 10:35 UTC: partial combined results

- First200-update original baseline: median76,877.45 TPS/GPU,3.409895s/update;
  first combined AR:97,633.10 TPS/GPU,2.684991s/update (+27.0%). Windows31–200,
  no skipped updates. CE means1.955127/1.954813. These are first-repeat results,
  not the final A/A numerical assessment. Combined direct RS is running near101k.
- Fresh smoke reference completed0->4->8 with held-out metrics; optimized saved
  step0/4, evaluated, and restored step4. All64 distributed initial-weight hashes
  match. Full batch/resume/upload sign-off is still pending.
- CPU audit now `01M1RH9SS78TSBTENE8R1Z9CF2` (source7c304206d), replacing the
  earlier collectors only to improve diagnostics/partial-eval handling. No uploader
  or training restart for these collector changes. Superseded CPU collectors stopped.
- Read-only checkpoint immutability audit `01M1RHYJYQ0WJKR78JDWZXKEYA` passed:
  all1,092 recorded files each for reference0/4 and optimized0 still have matching
  sizes/mtimes after resume. Reference4 remotely verified on first attempt at10:30:11,
  after starting10:10:58. Optimized0 verified10:32:21. Uploader high-performance
  mode is enabled (8CPU,32GiB,node-local Xet cache); do not treat the much faster
  step0 transfer as physical bandwidth, since deduplication can dominate it.
- Final light combined-RS Nsight capture queued:
  `01M1RJ0FGZ2M9CY12VSGKSGR5M`, source7c304206d,64GPUs,100updates,
  standalone2026.4.1, ranks0/8/.../56, capture71–73, autograd NVTX off.
  Collector `01M1RJ3VFYZFKTWQ9BT85DQ32J`; only small summaries go to results.
- Rechecked final serialized configs after the GCS/RS changes: exact model,
  dataset/data-loader, eval and optimizer/scheduler config equality; only the
  intentional `dp_config.use_reduce_scatter` field differs. Retention never,
  no max-checkpoint removal, sync saves, MB4, no AC/FP8 all asserted locally.

## September 5, 11:03 UTC: production smoke signed off

- All eight workers of corrected smoke `01M1RFHFA1GFSQMDN4Q3QRQ15B` exited0.
  Both arms trained0->4, restored model/Adam/trainer in a separate process, then
  trained4->8 and evaluated. All64 initial-weight fingerprints and128 first/resumed
  batch fingerprints per arm match. Partial smoke evaluation had72 finite observed
  metric records and16 explicitly unmeasured records per arm; final runs evaluate
  the complete validation epoch, not this two-batch smoke subset.
- Audit `01M1RH9SS78TSBTENE8R1Z9CF2` exited0 at10:56:28; all eight gates true.
  All six step0/4/8 checkpoints were remotely verified on their first upload attempt,
  and both current manifests were published and read-back verified. Latest upload
  verification10:56:20. Each checkpoint has1,092 files and~150.055GB. No deletion.
  Weka free space39.346TB, versus~7.50TB required for both full runs.
  Small audit dataset `01M1RH9SSHQSNCYQ5KGGR2RKFX`, local
  `profiling-analysis-20260905/integration-smoke-signoff-r3`.
- First combined direct-RS repeat finished:100,515.67 TPS/GPU,2.607991s/update,
  +30.75% versus its76,877.45 reference; AR97,633.10 (+27.0%). Median updates31–200,
  all200 loss/norm records finite and no skipped steps in any of these three arms.
  RS/reference CE mean absolute difference.00130818, maximum.00510883. However,
  maximum absolute20/50/100-step mean differences.00290349/.00124829/.00060260
  exceed the corresponding earlier A/A windows. Do not call the trajectories
  bit-identical or declare every numerical statistic within the old A/A envelope.
  Second repeats and the planned100B comparison are required follow-up evidence.
- Final combined-RS Nsight `01M1RJ0FGZ2M9CY12VSGKSGR5M` completed100 updates;
  collector `01M1RJ3VFYZFKTWQ9BT85DQ32J` exited0. All eight sampled CUDA traces
  valid. Uncaptured31–70 median98,231 TPS/GPU; post-capture81–100 median94,627.
  Instrumentation/allocation differences make this inappropriate to replace the
  same-allocation unprofiled A/A timing. Dataset `01M1RJ3VGA18NGJN520AAXXEAP`.
  Read-only timeline-union analysis `01M1RKPRPZWNJNAG6VXP69EATB` also exited0;
  no raw traces/checkpoints are exported to Beaker results.

### Residual headroom, not additional approved gains

The final three-update capture has99,359 kernels per sampled rank and9.35–10.15s
kernel spans. Union-of-intervals analysis finds2.04–3.12s of collective activity
without another recorded kernel and0.19–1.08s with no recorded GPU operation.
Those are instrumented observations, not additive/removable critical-path budgets:
capture perturbs launch timing and collectives include cross-rank waits. Rank0
has one644ms uint64 collective versus~10ms for the other two instances; its cause
is not isolated here, so do not extrapolate that outlier as steady-state overhead.

Gross kernel sums per captured update, across the eight sampled ranks:

| Area | Approximate kernel time/update | Decision for this integration |
|---|---:|---|
| Grouped expert GEMMs including rounded wgrad |572–586ms|Keep qualified rounded-wgrad; activation fusion rejected as-is|
| KDA-family kernels |355–364ms|Keep qualified CTA policy; WY scheduling probes did not win|
| Strict-FP32 router GEMMs |228–239ms|BLAS preference probe no gain; no precision change|
| Paired SwiGLU backward |94–96ms|Keep qualified implementation|
| Native-tie top16 |34ms|Keep qualified implementation|

These kernel sums overlap other streams and must not be added to the interval
unions or interpreted as attainable speedups. Future work could target expert
projection/activation fusion, KDA intermediate traffic, router GEMMs, or optimizer
communication, but none has another qualified improvement ready for this bundle.
Freeze the current candidate after its repeats rather than adding an unmeasured
last-minute switch. Detailed timeline dataset `01M1RKPRQ9311VN6Z3QT3HBPHN`.

## September 5, 11:24 UTC: final short-run qualification

All eight workers of `01M1RERW7CSCBHWKDDGEED3AFT` and collector
`01M1RF3AX73HKHV3HS7VRCHD7M` exited 0. All six arms contain exactly 200 finite
CE/norm updates, steps 7501–7700, with zero skipped updates (1,200 updates total).
Dataset `01M1RF3AXH0PN8SN64XX5NCSPG`; local
`profiling-analysis-20260905/combined-aa200-r1`.

| Arm | Repeat | Median TPS/GPU | Median TFLOPs/GPU | Median seconds/update |
|---|---:|---:|---:|---:|
| Original baseline |1|76,877|335.10|3.409895|
| Original baseline |2|76,926|335.31|3.407748|
| Combined, all-reduce |1|97,633|425.58|2.684991|
| Combined, all-reduce |2|97,978|427.08|2.675541|
| Combined, direct reduce-scatter |1|100,516|438.14|2.607991|
| Combined, direct reduce-scatter |2|101,549|442.65|2.581451|

Timing uses median updates 31–200, excluding compilation/warm-up. Same allocation,
same checkpoint/data, interleaved repeats. Selected RS gains are **30.75%/32.01%**
over their corresponding references, not a multiplication of individual gains.

| Loss difference over all 200 updates | Mean absolute | Maximum absolute | Max absolute rolling mean, 20 / 50 / 100 updates |
|---|---:|---:|---|
| Baseline A/A |0.0013690|0.0067981|0.0030669 /0.0015430 /0.0006711|
| Combined AR A/A |0.0011448|0.0058104|0.0016575 /0.0005822 /0.0003767|
| Combined RS A/A |0.0012270|0.0049865|0.0011117 /0.0005560 /0.0001915|
| RS minus baseline, repeat1 |0.0013082|0.0051088|0.0029035 /0.0012483 /0.0006026|
| RS minus baseline, repeat2 |0.0013322|0.0066297|0.0015583 /0.0005535 /0.0003538|

The newly repeated original baseline shows larger short-window excursions than
the earlier pre-paired reference A/A, resolving the earlier concern about that
comparison envelope. The selected RS/baseline loss summaries above are within
this new observed baseline A/A envelope. This is not proof of bitwise identity or
long-horizon equivalence. Norm mean absolute differences are 0.01029/0.01168 versus
baseline A/A 0.01155; maximum differences 0.04517/0.05656 versus A/A 0.06548. Thus
do not claim every norm statistic is strictly bounded by the A/A result either.

Decision: freeze `core-docpool-top16-wgrad-rs` for the authorized 100B experimental
comparison. Primitive/optimizer tests, repeated training, and production-scale
fresh save/restore/eval/upload smoke passed. Retain all noted rounding/outlier
caveats. No additional kernel, precision, architecture or parallelism switch.
Broad deployment still requires reviewing the long-run results.
