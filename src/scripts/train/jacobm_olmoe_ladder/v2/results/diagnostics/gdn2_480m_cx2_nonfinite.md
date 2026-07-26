# Canonical GDN2 480M Cx2 non-finite diagnosis

Date: 2026-07-26

## Target

- Run: `pt-480m-geometry-hybrid-gdn2-ev1-noneg-nope-gated-cx2-lr9e-4-r1`
- Architecture: canonical GDN2, `expand_v=1`, negative eigenvalues disabled,
  NoPE, gated attention
- Source checkpoint: `step24500`
- Repeated failure: step `24668` in all seven prior resumes

## Exact replay

The [read-only localizer
run](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYFSKK0582424EVT7YAY1WHK)
restored the model, optimizer, scheduler, dataloader, and RNG state from
`step24500`. Checkpoint writes, W&B, and eval callbacks were disabled. It
reproduced the non-finite loss at exactly step `24668`.

All checkpoint model parameters were finite. Only rank 0 had a non-finite
local forward/loss before the loss all-reduce. The first bad module boundary
was `module.blocks.0.attention`, the first GDN2 layer, for rank-local sequence
index 3. The captured layer input and all 17 layer parameter tensors were
finite.

The first bad production output was token `7104`; all 768 output channels were
NaN from tokens `7104..8191`. Since `7104 = 111 * 64`, the production failure
appears at an exact GDN2 chunk boundary.

Exact captures are under:

`/weka/oe-training-default/ai2-llm/checkpoints/jacobm/olmoe3/olmo-ddp/debug/gdn2-480m-cx2-localizer/canonical-gdn2-480m-cx2-step24668-r1/`

## Production chunk versus sequential FP32 reference

The successful [one-B300 reference
comparison](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYFV6TYHTH55BQJC2G6XJAME)
recomputed the exact captured layer invocation with the pinned production FLA
commit `cbb0a72`, then ran FLA's token-by-token PyTorch GDN2 recurrence with
FP32 state accumulation.

Both paths diverged in the same recurrent head/channel:

| Path | First non-finite token | Head | Channel | Largest finite raw value |
|---|---:|---:|---:|---:|
| Sequential FP32 recurrence | 7039 | 2 | 105 | `2.93e37` |
| Production 64-token chunk | 7104 | 2 | 105 | `6.48e37` |

Both final states contained exactly 16,384 NaNs, corresponding to the entire
`128 x 128` state for head 2. The optimized output reproduced the captured
failure at token 7104. On their finite overlap, captured versus recomputed
production output had cosine `0.99999976` and relative L2 `6.73e-4`.

This rules out a Triton-only chunk-kernel bug as the root cause. The learned
GDN2 recurrence itself becomes numerically/dynamically unstable for this
sequence even under the sequential FP32 reference; chunking merely delays the
visible failure to the next chunk boundary.

## Offending sequence

The bad sequence is from `stack-edu_Cpp` and is an extreme repetition outlier:

| Rank-0 sequence | Source | Unique tokens / 8192 | Most frequent token | Lag-64 matches |
|---:|---|---:|---:|---:|
| 0 | `s2pdf_software_development` | 722 | 584 | 209 |
| 1 | `all-dressed-snazzy2_health` | 2,519 | 328 | 69 |
| 2 | `all-dressed-snazzy2_entertainment` | 2,412 | 344 | 87 |
| 3 (bad) | `stack-edu_Cpp` | **12** | **2,340** | **2,294** |
| 4 | `s2pdf_science_math_and_technology` | 949 | 462 | 121 |
| 5 | `all-dressed-snazzy2_electronics_and_hardware` | 2,038 | 303 | 59 |

The current instance filter checks repetition periods only through 13, so the
longer repeated structure is not filtered.

## Conclusion and recommended follow-ups

1. Treat the present failure as a GDN2 state-stability problem exposed by a
   pathological long repetitive example, not as evidence that the optimized
   forward kernel computes the wrong recurrence.
2. A synchronized skip-step can safely get a run past this isolated batch
   because the non-finite is detected before the optimizer update, but it is a
   mitigation rather than an architectural fix and similar examples may recur.
3. Test the least invasive data mitigation independently: extend the
   repetition filter to longer periods (at least 64) or explicitly reject this
   extreme instance, then replay from `step24500` and verify the trajectory.
4. In parallel, discuss an architectural stability guard for GDN2 (bounded or
   normalized recurrent state/update). This needs a controlled 275M sweep;
   arbitrary state clamping should not be inserted into current results.

The first analyzer attempt
([job](https://beaker.org/orgs/ai2/workspaces/OLMo-3-moe-experiments/work/01KYFTP7RXFP70VNA8R55JFQJ4))
completed both recurrences but failed in analysis-only post-processing because
the FP32 reference output was passed directly to a BF16 projection. The
successful rerun retains FP32 for raw/state comparisons and casts only at the
model's projection boundary.
