# Landmark top-k TRAINING — design + cost model

Goal: train with hard top-k landmark retrieval so that training is **drastically faster** than the
current dense landmark training, and so that the trained model is natively matched to top-k
inference instead of being a dense model that top-k is bolted onto at eval.

## The one decision that determines everything: selection granularity

Inference top-k (both the shipped decode and `landmark_prefill_topk`) selects **per query token**.
That cannot be made fast in a training kernel: a Triton program handles a tile of 64 queries at once,
and if each row selects its own blocks the tile must load the *union* of 64 rows' selections — which
approaches dense. Per-token selection is why `landmark_prefill_topk`'s fused path is still O(T²) (it
masks rather than skips).

So training must select **per query block**: all 64 queries in a block share one set of `k` key
blocks, scored by pooling the block's queries against each landmark key (mean or max over the 64
rows — mean is the safer default, max is more permissive). This is the same choice NSA and MoBA make,
and it tiles perfectly: one indexed pointer walk over `k` blocks per program, no masking, no union.

Consequences to be explicit about:

* It is an **approximation of, not the same thing as, the inference-time rule**. Two options at eval:
  keep per-token selection (train/inference mismatch, but the mismatch is in the *permissive*
  direction — inference gets a strictly better-matched set) or switch inference to block-granular
  too. Worth measuring both; `landmark_prefill_topk` already gives us the per-token eval number to
  compare against.
* Selection is **non-differentiable**. Gradients flow to the selected blocks' landmark scores through
  the gate softmax (exactly as in the dense grouped softmax); non-selected blocks get none. This is
  standard hard routing — same as top-k MoE — and it is why the model can still *learn* which blocks
  to retrieve: a selected block that turns out useless has its gate score pushed down, letting
  another block win next time.
* A block that is never selected early in training gets no gradient and may stay unselected. MoE
  fixes this with load-balancing losses; the cheap analogue here is to keep a small `alpha` reserve
  on non-selected landmarks (the compressive variant already has one) so every block keeps a gradient
  path. **The 4B eval says alpha buys nothing at inference — but that is a different claim from
  whether it matters during training**, and it is the obvious first ablation.

## Cost model

Per query token, keys touched:

| | dense causal | dense landmark | block top-k landmark |
| --- | --- | --- | --- |
| keys | `T/2` | `T/2` | `T/B + (k+1)·B` |

Ratio vs dense, `B = 64`:

| T | k = 10% of blocks | k = 8 | k = 16 |
| --- | --- | --- | --- |
| 8k | 4.3x | 5.8x | 4.3x |
| 16k | 4.3x | 9.8x | 6.6x |
| 32k | 4.3x | 15.1x | 9.4x |
| 64k | 4.3x | 21.3x | 15.1x |

Two things fall out:

1. A **percentage** budget caps at `~1/f` (4.3x at 10%) no matter how long the context — `k` grows
   with `T` so the `k·B` term keeps pace. A **fixed** `k` is what compounds with context length.
   This is the same conclusion the accuracy sweep reached from the other direction (cost tracks
   absolute blocks kept, not the fraction), which is a good sign the two agree on what to optimize.
2. The floor is the landmark scan (`T/B` per query, i.e. `1/B` of dense), so **no version of this
   beats `B = 64`x**, and you only approach that with small `k` at long `T`.

At our 32k training length with `k = 8`, the arithmetic ceiling is ~15x on the attention. Attention is
not all of a training step — for Qwen3-4B at 32k, MLP+projection FLOPs are comparable to attention —
so expect the *step* speedup to be well under the attention speedup. That number should be measured,
not assumed, and it is the first thing the prototype below reports.

## Implementation plan

The existing training kernels are already 90% of the way there. `_fwd_kernel_compressive` loops
`for start_n in range(0, N_PREFIX_Q + start_m)` over every past key block; the sparse version loops
over `k` **indexed** blocks instead, reading the block id from a `(Z, H, n_qblocks, k)` int32 tensor
and doing the pointer arithmetic from it. That is a small, local change.

The backward is where the work is:

* `_bwd_q_kernel` — one program per query block, loops over that block's key blocks. Same indexed
  walk as the forward. Easy.
* `_bwd_kv_kernel` — one program per *key* block, loops over the query blocks that attend it. Under
  sparsity this needs the **inverse index** (for each key block, which query blocks selected it),
  which is ragged. Two ways out: (a) build a CSR-style inverse index on the host each step, or
  (b) drop the atomic-free design and accumulate `dk`/`dv` with `tl.atomic_add` from a query-block-
  parallel kernel. (b) is much less code but gives up bitwise determinism, which the current kernels
  deliberately preserve — call that out before choosing.

Order of work:
1. **Forward kernel + benchmark** — establishes the real (not analytic) attention speedup and the
   end-to-end step speedup ceiling. Cheap, and it tells us whether the rest is worth it.
2. Backward (`dq`, then `dk`/`dv`).
3. Gradient check against the dense kernel restricted to the same selection (the selection makes them
   *equal*, not merely close, when `k >= n_blocks` — that is the strongest available test).
4. A short training run vs the dense baseline: loss curve match at `k = n_blocks`, then the real
   `k = 8` run.

Step 1 is what the sibling `bench_train_fwd.py` measures.
