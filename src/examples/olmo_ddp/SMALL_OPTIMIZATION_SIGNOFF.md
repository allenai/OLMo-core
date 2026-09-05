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
