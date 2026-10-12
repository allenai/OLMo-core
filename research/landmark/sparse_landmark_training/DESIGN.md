# Landmark top-k TRAINING — design + cost model

Goal: train with hard top-k landmark retrieval so that training is **drastically faster** than the
current dense landmark training, and so that the trained model is natively matched to top-k
inference instead of being a dense model that top-k is bolted onto at eval.

## The one decision that determines everything: selection granularity

Inference top-k (both the shipped decode and `landmark_prefill_topk`) selects **per query token**.
The naive training kernel cannot do that efficiently: a Triton program handles a tile of 64 queries
at once, and if each row selects its own blocks the tile must load the *union* of 64 rows' selections
— which approaches dense. Per-token selection is why `landmark_prefill_topk`'s fused path is still
O(T²) (it masks rather than skips).

> ⚠ **Corrected 2026-07-30 after reading the papers.** An earlier version of this doc claimed
> per-query-*block* selection "is the same choice NSA and MoBA make." That is wrong on both counts.
> **Neither pools the selection over query positions.** NSA selects per query position and shares the
> selection across the query heads *within a GQA group*; MoBA selects per query token and solves the
> tiling problem by inverting the loop. Do not cite them as precedent for block-granular query
> pooling.

Three real options, two of them with published precedent:

**(a) NSA — share selection across the GQA group.** NSA keeps per-position selection but forces every
query head in a GQA group to use the same blocks: *"consistent block selection across heads in a group
has to be ensured to minimize KV cache loading during decoding"*, implemented by summing the
per-head importance scores over the group. The kernel then puts the **query loop in Triton's grid**
and the selected-block loop inside, so each program serves one query position × all heads of its
group. Reported at 64k: **9.0x forward, 6.0x backward, 11.6x decode**, with selection block 64,
n = 16 selected blocks, sliding window 512.

⚠ The catch for *us*: this makes the per-program query tile `n_rep × head_dim`. NSA's group is large
enough for that to fill a `tl.dot`; **Qwen3-4B is 32 q-heads / 8 KV heads, so `n_rep = 4`** — a 4×128
tile, below the 16-row minimum `tl.dot` wants. NSA's kernel shape does not transfer to our head
config without padding or batching query positions (which reintroduces the union problem).

**(b) MoBA — invert the loop.** Keep per-token selection and reorganize the computation around key
blocks instead of query tiles: (1) route each query to its top-k blocks, (2) **sort queries by
assigned block**, (3) run varlen FlashAttention per block over just the queries routed to it,
(4) un-sort, (5) combine each query's partial outputs with **online softmax**. No union, no pooling,
and it reuses an existing varlen kernel. Reported 6.5x prefill at 1M tokens, 16x attention time at
10M. MoBA's blocks are much coarser than ours (512 / 2048 / 4096 with top-k 3–12) — at 32k their
budget is 2048×3 ≈ 6k tokens, comparable to our 10%-of-blocks ≈ 3.3k, so we are in the same sparsity
regime with finer granularity.

Our grouped softmax is compatible with step (5): the gate normalizes over *selected landmarks +
local*, which spans blocks, but that is exactly a streaming softmax — the existing fused landmark
kernel already maintains running `m`/`l` across key blocks, so the online-combine machinery is
already written, just not in this arrangement.

**(c) Pool the selection over each query block** (the original proposal here): all 64 queries in a
block share one set of `k` blocks, scored by pooling their landmark scores. Tiles perfectly and is
the least kernel work, but it has **no precedent in either paper** and is the largest modeling change
of the three. Treat it as the fallback if (a) and (b) both prove awkward, not the default.

**Recommendation: adopt NSA's GQA-group sharing regardless of which kernel shape wins.** It is a
small modeling change, it is validated at scale, and it directly fixes a measured problem in our own
code: the decode gather in `landmark_sparse_decode` re-reads rows for each of the 4 heads in a group,
which is exactly why k=51 only reached 2.5x on key-bytes while k=8 reached 16x. Group-shared
selection removes that `n_rep` factor from both decode and training.

Consequences of any hard selection, to be explicit about:

* If training and inference use different selection rules, that is a **train/eval mismatch**. Option
  (c) has it by construction; (a) removes it only if inference also shares across the group (it
  currently does not); (b) has none. `landmark_prefill_topk` already gives us the per-token eval
  number to compare any variant against.
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
* **Do not make every layer sparse.** MoBA's production recipe keeps the **last three layers as full
  attention** and switches the other 29, and it activates sparsity as a *continued-pretraining phase*
  (100B tokens after the dense long-context extension) rather than training sparse from scratch. That
  rhymes with our own hybrid full/chunked-layer finding, so a full-attention tail is the default to
  start from, not an afterthought.

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

**Calibration against a real kernel.** NSA runs almost exactly our configuration — selection block 64,
n = 16 — and reports **9.0x forward / 6.0x backward at 64k**. The table above predicts 15.1x for
`k = 16` at 64k. So a well-tuned production kernel lands at roughly **60% of this analytic ceiling on
the forward and 40% on the backward**, and the backward is the weaker of the two. Budget against those
factors, not against the table: at our 32k / `k = 8` the honest expectation is ~9x forward and ~6x
backward *on the attention*, before amortizing over the MLP.

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

## References

* NSA — *Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention*,
  DeepSeek-AI / PKU / UW, arXiv [2502.11089](https://arxiv.org/abs/2502.11089).
  Compression + selection + sliding-window branches with a learned gate; selection scores derived by
  blockwise aggregation of the *compression* branch's attention scores and **summed across heads in a
  GQA group**; Triton kernel with the query loop in the grid. l = 32 / d = 16 / l' = 64 / n = 16 /
  w = 512. 9.0x fwd, 6.0x bwd, 11.6x decode at 64k.
* MoBA — *Mixture of Block Attention for Long-Context LLMs*, Moonshot AI, arXiv
  [2502.13189](https://arxiv.org/abs/2502.13189), code
  [github.com/MoonshotAI/MoBA](https://github.com/MoonshotAI/MoBA).
  Gate `s_i = <q, mean_pool(K[block_i])>`, per-token top-k, current block forced in with a causal
  mask and future blocks set to `-inf`; implemented by sorting queries by assigned block, varlen
  FlashAttention per block, un-sort, online-softmax combine. Block 512–4096, top-k 3–12. 6.5x prefill
  at 1M, 16x attention time at 10M. Trained as a continued-pretraining phase with the last 3 layers
  left as full attention.
