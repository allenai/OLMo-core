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

### `qblock` — approximate

Pool the 64 queries' landmark scores into one top-k per query block, so the list is exactly `k` long.
Always fast, regardless of agreement.

| path (32k, k=8) | s/prompt | vs landmark kernel | vs dense flash |
| --- | --- | --- | --- |
| SPARSE qblock | **0.42** | 3.4x | 2.1x |

## Accuracy cost (Qwen3-4B, contradiction v2, eval_size=500, SE ±0.019–0.022)

| prefill | 2k | 8k | 16k | 32k |
| --- | --- | --- | --- | --- |
| baseline (dense prefill, decode-only top-k) | 0.783 | 0.741 | 0.626 | 0.554 |
| exact per-token top-10% (masked or `union`) | 0.768 | 0.742 | 0.618 | 0.552 |
| **`qblock` top-10%** | **0.721** | **0.685** | **0.583** | (running) |

`qblock` costs about **0.05 f1** — 2–3 SE, so a real regression, not noise. Pooling the selection over
64 queries is not free: a token that needed a block its neighbours did not want loses it.

## Where this lands

* **`union` is the better trade**: exact by construction (no accuracy run needed, ever) and the
  measured 38% coverage implies ~2.6x over the dense landmark kernel — roughly 1.8x *under* `qblock`'s
  3.4x, but without giving up 0.05 f1.
* `qblock` is the fallback when speed dominates and the loss is acceptable.
* If exact-and-faster is wanted, the next step is MoBA's loop inversion (sort queries by assigned
  block, varlen-attend per block, recombine with online softmax), which keeps per-token selection and
  does not pay the union's overhead at all. More kernel work, no modelling change.
