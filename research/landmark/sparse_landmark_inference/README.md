# Making landmark inference actually fast

Companion to `debug/prefill_topk/` (which answered the *accuracy* question). This one is about speed.

## Starting point: the shipped landmark decode is 19x slower than dense attention

Profile at 32k, Qwen3-4B shapes (32 q-heads / 8 KV heads, D=128, 36 layers), per generated token:

| | ms/token | GB/token |
| --- | --- | --- |
| dense SDPA, GQA, no expansion (what a normal model pays) | **1.8** | 4.50 |
| `repeat_kv` expansion **alone**, no attention at all | 20.0 | 22.50 |
| landmark decode, dense gating (as shipped) | 31.1 | 22.50 |
| landmark decode, top-51 (as shipped) | **34.8** | 22.50 |

Two separate problems, both fixed in `landmark_sparse_decode.py`:

1. **`repeat_kv` is 57% of the step.** `_forward_generate` expands the KV cache from 8 heads to 32
   every step of every layer via `expand().reshape()` — and a reshape of a stride-0 expand *cannot*
   be a view, so it materializes a 4x copy of the whole cache before a single score is computed.
   Folding the query into the KV-head grouping instead removes it entirely.
2. **Top-k was numerical, not computational.** The shipped decode scores all `T` keys and masks the
   losers, which is why top-51 is *slower* than dense gating (34.8 vs 31.1) — masking is extra work
   on top of the same scan. The sparse version scores only the `n_blocks` landmark keys, then
   gathers just the selected blocks plus the local section.

## Result

Same outputs (validated to bf16 noise against the shipped decode across 21 cases: plain/compressive,
alpha 0/0.1, `top_k` 1..None, eval and per-block prompt decode, landmark and content query positions,
GQA and MHA, batch 1 and 3), per token at 32k:

| | ms/token | GB/token | vs shipped |
| --- | --- | --- | --- |
| landmark decode top-51 (shipped) | 34.8 | 22.50 | 1.0x |
| sparse decode k=51 | 11.1 | 1.87 | **3.1x** |
| sparse decode k=8 | 10.9 | 0.36 | **3.2x** |

Memory traffic drops **62x** (22.50 → 0.36 GB/token at k=8). Wall clock only drops 3.2x, and the
reason is visible in the table: **the sparse time is flat at ~11 ms across 8k/16k/32k and flat across
k**. It is no longer memory-bound — it is bound by kernel-launch and Python-op overhead. At 0.30
ms/layer for ~10 MB of traffic we are ~100x off the bandwidth roofline.

So the honest status: **sparse decode is 3.2x faster than the shipped landmark decode, but still ~6x
slower than plain dense flash-decode** (1.8 ms), because it is ~20 small torch ops per layer against
SDPA's one fused kernel.

## To beat dense attention: fuse it

The traffic is already 12x below dense SDPA's; only the op count is in the way. Plan:

* stage A stays in torch — `matmul` against the landmark keys, then `topk`. 2 launches.
* stage B becomes **one** Triton kernel: given the selected block ids, load those blocks' K/V plus
  the local section, do the gate/within grouped softmax and the AV accumulation in registers, write
  the output. 1 launch.

3 launches per layer instead of ~20 should land near the bandwidth bound (~2 ms/token at 32k),
i.e. at or below dense SDPA — and unlike SDPA it stays *flat* as context grows, so the margin widens
with length. That is the piece not yet written.

## Files

| file | what |
| --- | --- |
| `../../src/olmo_core/nn/attention/landmark_sparse_decode.py` | implementation + `enable_sparse_decode(model)` |
| `test_sparse_decode.py` | equality vs the shipped decode, 21 cases |
| `bench_decode.py` | the tables above |

⚠ A bug worth remembering: the first version silently dropped `softmax_scale` on the local-section
scores only. Every output was wrong by a factor of √D **in one term**, which looks like a subtle
numerical issue rather than a missing multiply. The gate weights and the gather were both provably
correct in isolation — it only showed up when the reference's per-block probability *sums* were
compared against the gate, so build that decomposition into the next debug session too.
