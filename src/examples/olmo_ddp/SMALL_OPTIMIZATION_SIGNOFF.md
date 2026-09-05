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
