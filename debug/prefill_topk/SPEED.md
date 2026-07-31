# Making landmark top-k prefill actually fast

`landmark_prefill_topk` answered the accuracy question but was the **masked** form: it scores every
past block and floors the losers out of the gate softmax. Cost is therefore independent of `k`, and
top-k made prefill *slower* than not using it.

Per prompt at 32k (Qwen3-4B shapes: 32 q-heads, D=128, 36 layers):

| path | s/prompt |
| --- | --- |
| dense flash attention (SDPA) | 0.90 |
| dense landmark kernel (what prefill runs today) | 1.45 |
| masked top-k, k=8 (`landmark_prefill_topk`) | 2.50 |

## The fix: iterate a candidate list, not every block

`landmark_prefill_sparse.py` gives each query block a list of candidate key blocks and walks it with
a **data-dependent Triton loop bound**, so work scales with the number of selected blocks. One
kernel, two ways to build the list.

### `union` — exact

Each 64-query block iterates the **union** of its 64 rows' per-token top-k picks, and the per-row
cutoff is re-applied inside the kernel (same midpoint threshold as the masked path). Verified
**bit-identical** to the masked path.

Whether it is *fast* is an empirical question about how much neighbouring queries agree. Measured on
a real trained compressive-landmark model (`q06b-comp-contra-n20-sft-local`, 16k contradiction
prompt, 261 blocks, k=8, all 28 layers):

| | union size | vs k | covers |
| --- | --- | --- | --- |
| mean over layers | **33.9** | 4.2x | **38%** of each query's past blocks |
| best layer (12) | 21.5 | 2.7x | 27.8% |
| worst layer (0) | 60.6 | 7.6x | 58.0% |

So 64 queries × 8 picks = up to 512 selections collapse to ~34 distinct blocks — real attention heads
agree a great deal. **On random q/k the union covers 84–100% and buys nothing**, so this number could
only ever have come from a real model; do not benchmark this mode on synthetic tensors.

Early layers agree least (layer 0 is the worst by a wide margin), which is consistent with early
layers doing diffuse/positional mixing before the retrieval behaviour sharpens.

### `qblock` — approximate, and the one that wins

Pool the 64 queries' landmark scores into one top-k per query block, so the list is exactly `k` long.
Always fast, regardless of agreement.

**The gate score is LINEAR in the query**, so mean-pooling the 64 queries and dotting once is exactly
equal to dotting all 64 and averaging — at 1/64 the work. Building the candidate list was 82% of
qblock's runtime before this; after, it is 0.30 ms of a 0.67 ms total at 16k. (Same structural idea as
MoBA's `<q, mean_pool(K)>`, applied on the query side.) This one change took qblock from 2.5x to 8x
at 16k.

Per prompt, Qwen3-4B shapes (32 q-heads, D=128, 36 layers), after the pooled build:

| T | k | % of blocks | s/prompt | vs landmark kernel | vs dense flash |
| --- | --- | --- | --- | --- | --- |
| 32k | 8 | 1.6% | **0.088** | **16.5x** | **10.2x** |
| 32k | 16 | 3.1% | 0.128 | 11.4x | 7.0x |
| 32k | 32 | 6.2% | 0.208 | 7.0x | 4.3x |
| 32k | 51 | 10% | 0.302 | 4.8x | 3.0x |
| 32k | 128 | 25% | 0.635 | 2.3x | 1.4x |
| 16k | 8 | 3.1% | 0.042 | 9.0x | 5.5x |
| 16k | 64 | 25% | 0.172 | 2.2x | 1.3x |

(References at 32k: dense flash 0.90 s, dense landmark kernel 1.46 s, masked top-k 2.50 s.)

## Accuracy cost (Qwen3-4B, contradiction v2, eval_size=500, SE ±0.019–0.022)

| prefill | 2k | 8k | 16k | 32k |
| --- | --- | --- | --- | --- |
| baseline (dense prefill, decode-only top-k) | 0.783 | 0.741 | 0.626 | 0.554 |
| exact per-token top-10% (masked or `union`) | 0.768 | 0.742 | 0.618 | 0.552 |
| **`qblock` top-10%** | **0.721** | **0.685** | **0.583** | **0.509** |
| Δ (`qblock` − baseline) | −0.062 | −0.056 | −0.043 | −0.045 |

`qblock` costs **0.043–0.062 f1**. Each rung on its own is 2–3 SE, but all four move the same way by
a similar amount, so this is a real regression, not noise. Pooling the selection over 64 queries is
not free: a token that needed a block its neighbours did not want loses it. Note the loss does *not*
grow with context — it is a roughly constant offset, which is what you would expect if the damage is
per-query-block rather than cumulative.

## Where this lands

⚠ **`union` does not pay, and an earlier version of this file said it would.** The 38% coverage is
real, but coverage only bounds the *kernel*; the candidate-building dominates. Measured at 16k
(0.6B shapes), against a 5.47 ms dense attention: union's build alone is **5.02 ms**, so even a
zero-cost kernel leaves it at 0.92x. On real tensors it measured **0.73x — slower than dense**.
Extrapolating a speedup from a coverage fraction was the mistake; always measure the build too.

Union's build is expensive for a structural reason qblock's is not: selection is per token, so it
*needs* the full `(B, H, T, n_blocks)` score matrix (274 MB in fp32 at 16k) plus a top-k, a mask, and
a compaction over it. qblock needs only `(B, H, n_qblocks, n_blocks)` — 64x smaller. Rewriting union's
build (bf16 scores, cumsum compaction instead of argsort) could roughly halve it, but it still tops
out near 2x. Exact-and-fast really wants MoBA's loop inversion instead: sort queries by assigned
block, varlen-attend per block, recombine with online softmax — no union, no per-token score matrix.

**Recommendation: `qblock`, with `k` chosen by the accuracy/speed curve.**

| setting | speed at 32k | accuracy vs baseline |
| --- | --- | --- |
| qblock 25% (k=128) | 2.3x landmark / 1.4x flash | −0.012 (under 1 SE, free) |
| qblock 10% (k=51) | 4.8x landmark / 3.0x flash | −0.045 to −0.062 (2–3 SE, real) |
| qblock k=8 | 16.5x landmark / 10.2x flash | untested |

The accuracy sweep says the cost tracks **absolute** blocks kept, so a fixed `k` in the 32–64 range is
the obvious thing to test next: at 32k that is 7.0x/4.3x while keeping more blocks than the 10%
setting does at 8k and 16k, where the losses were just as large.
