# Chunked-eval speedup, round 2 — measured at 8k, not 2k

Round 1 (2026-08-04, `records/paper-v2-todo-status.md`) closed with a negative result: the
`_build_chunk_ids_for_batch` rebuild was 0.4% of runtime, `create_block_mask` was already
`torch.compile`d, and the BLOCK_N pow2 clamp moved the number by 1.6%. All three of those
measurements were taken at the **2k rung**, where a prompt prefills in one step and the KV cache
sits at its 128-page floor.

This round re-measured at **8k**, where the production numbers actually hurt.

**Headline:** the chunked arm is not GPU-bound and the mask machinery is not the main cost. At 8k
the GPU is idle roughly two thirds of every forward step, and the run does ~50x more forward steps
than dense because chunked mode is capped to `max_num_seqs=8` while dense runs uncapped.

**Bottom line — validated, not extrapolated:** `CHUNK_CACHE_IDS=1` + `max_num_seqs=32` (leaving
`max_num_batched_tokens` at its production 2048) gives **1.78x** on the full 500-example 8k
contradiction rung — **2066.1s → 1163.9s** — at `set_f1` 0.7999 vs the known-good 0.8039
(Δ −0.0040, SE 0.018) and `parse_rate` 1.000. Two env vars, no code path changed, score reproduced.

That is a real dent but not the whole gap: 17.8x → 10.0x vs dense. §1 explains why no further config
change gets there — ~50% of runtime is per-step Python outside the metadata builder, and CUDA graphs,
the standard tool for that, crash on this code path (§2c). Nothing has been applied; see §4.

## 0. The gap is 17.8x, not 7x

From the production 8k contradiction runs on `ctc-4b-contradiction-cmix`, `eval_size=500`:

| arm | gen_seconds | set_f1 | parse_rate | builder calls |
|---|---|---|---|---|
| dense (`--mode full`) | 116.4 | 0.9760 | 1.000 | — |
| chunked | 2066.1 | 0.8039 | 1.000 | 32573 |

(`/data/prasann/ctc_suite_vllm_results{,_chunked}/contradiction_iid/grade_8192.json` on cubbins.)

**17.8x.** Two structural facts fall straight out of that table:

1. `patch_debug` reads `{"direct": 0, "fallback": 32573}` — every single builder call takes the
   fallback `create_block_mask` path. The `_build_block_mask_direct()` branch is confirmed dead
   code in production (see §5).
2. 32573 builder calls for 500 examples is **~521 forward steps per example**, i.e. essentially
   every example runs to the `max_new_tokens=512` cap and the chunked arm only ever has 8 of them
   resident. Dense runs the same 500 examples uncapped (`extra={}` → vLLM's default
   `max_num_seqs=1024`). Chunked does roughly **50x more forward steps** than dense and is only
   17.8x slower, so its steps are individually *cheaper* — the gap is dominated by step count, not
   by the flex kernel being slow per unit of work.

## 1. Per-stage attribution at the 8k rung (the round-1 gap)

`debug/chunked_eval_speedup/tile_sweep.sbatch`, arm `arm00_profile`, 24 examples on
`eval_rungs/contradiction_iid/rung_8192.jsonl`, `ctc-4b-contradiction-cmix`, production config.
1578 builder calls = **100 prefill steps + 1478 decode steps**. Profiling cost 3.3% of wall clock
(115.0s profiled vs 111.3s unprofiled), so the attribution is not distorting the thing it measures.

| stage | seconds | % of 115.0s | calls | ms/call |
|---|---|---|---|---|
| `sync_head` — *all GPU work, drained* | 37.07 | 32.2% | 1578 | 23.49 |
| `build_block_mask` (decode) | 7.05 | 6.1% | 1478 | 4.77 |
| `build_block_mask` (prefill) | 3.70 | 3.2% | 100 | 37.04 |
| `orig_build` (prefill) — vLLM's own | 3.03 | 2.6% | 100 | 30.26 |
| `orig_build` (decode) — vLLM's own | 2.99 | 2.6% | 1478 | 2.02 |
| `build_chunk_ids` (decode) | 2.93 | 2.5% | 1478 | 1.98 |
| `build_chunk_ids` (prefill) | 0.20 | 0.2% | 100 | 2.03 |
| `make_mask_mod` | 0.00 | 0.0% | 1578 | 0.003 |
| **attributed** | **56.97** | **49.5%** | | |
| **unattributed** (scheduler, eager forward launch, sampler, detokenize) | **~58.0** | **~50.5%** | | |

### How to read this — and a correction to the round-1 method

`_prof_add(..., sync=True)` synchronizes *before* stopping the clock, so a stage timed that way is
billed for every kernel still in flight when it started. `build_chunk_ids` was timed that way, which
is why round 1's follow-up profile reported it at 24.9 ms/call with 8 resident sequences but
0.75 ms/call with 2 — a 33x swing for a 4x change in work. It was measuring the *previous step's
model forward*. A pure-numpy microbenchmark of the identical function
(`bench_chunk_ids_r2.py`) gives 4.65 ms at 8 reqs and 1.18 ms at 2, linear in `num_reqs`:

```
 num_reqs   ms/call        (seq_len=8875, n_chunks=187)
        1     0.638
        2     1.179
        4     2.346
        8     4.647
       16     9.344
       32    18.502
```

The fix used here is to drain the queue once at the top of the step under its own label
(`sync_head`) and time the CPU stages without syncing — `CHUNK_PROFILE_SPLIT=1`, which also splits
every stage into `.p` (prefill) and `.d` (decode) because the two have unrelated cost structures.

### The conclusion the numbers force

**115.0 s / 1578 steps = 72.9 ms per forward step, of which only 23.5 ms is GPU.** The chunked arm
spends ~68% of every step with an idle GPU, waiting on Python. Concretely:

* The whole chunked-mask apparatus — building chunk ids, swapping `mask_mod`, rebuilding the
  BlockMask — is **12.1%** of runtime (13.88 s). Even making it *free* buys 1.14x. Round 1's verdict
  that the mask machinery is not where the time goes **survives re-measurement at 8k**; it was
  right for the wrong reason (it measured a stage that was really absorbing GPU time).
* The dominant term is per-step *fixed* cost — ~50% unattributed CPU plus vLLM's own `orig_build` —
  which is paid once per forward step regardless of how many sequences that step advances.

That makes the two levers worth pulling **(a) fewer steps** (raise `max_num_seqs`) and **(b) less
Python per step** (CUDA graphs). Neither has anything to do with the mask.

## 2. Config changes tried

All arms: 8k rung, `ctc-4b-contradiction-cmix`, chunked, `--max-new-tokens 512`, greedy. Correctness
is checked two ways — `set_f1` / `parse_rate` from `general/grade_any.py` (the same scorer the
native harness uses), and `gen_match`, the count of generations byte-identical to the baseline arm.
`gen_match` is the sharper instrument here: at `eval_size=24` the binomial SE on f1 is ±0.088, far
too coarse to certify a config, whereas greedy decoding should reproduce the baseline exactly
unless the numerics changed.

### 2a. Tile sizes — REFUTED, and the intuition was backwards

`debug/chunked_eval_speedup/tile_sweep.sbatch` (job 3437991), `eval_size=24`.

| arm | config | gen_s | vs base | set_f1 | parse_rate | gen_match |
|---|---|---|---|---|---|---|
| `arm00_base` | production (kv=16, q=16, eager) | **111.3** | 1.00x | 0.7500 | 1.000 | 24/24 |
| `arm00_profile` | + `CHUNK_PROFILE_SPLIT=1` | 115.0 | 0.97x | 0.7500 | 1.000 | 24/24 |
| `arm04_q128` | `flex_attn_q_block_size=128` | 274.7 | **0.41x** | 0.7556 | 1.000 | 22/24 |
| `arm01/02/03/05/06` | `kv_block_size` 32 / 64 / 128 | — | — | — | — | blocked, see below |

**The starting hypothesis for this arm was wrong.** vLLM's `get_kernel_options` on the fallback
branch sets `BLOCK_M = min(64, q_block_size)` and `BLOCK_N = min(64, kv_block_size)`, so with the
production `q_block_size = kv_block_size = 16` the triton flex kernel runs **16x16 tiles** on an
H200 — which looks like an obvious pathology. It is not. Widening q to 128 (BLOCK_M 16 → 64) made
the run **2.5x slower**.

The reason is that block size is not just a tile size, it is the *resolution of the sparsity*. A
128-query block spans ~3 documents at this rung (8342 tokens / 187 chunks ≈ 45 tokens per doc), so
its kv-block set is the union of three documents' blocks and almost nothing can be skipped;
16-query blocks sit inside a single document and skip nearly everything. **The small tiles are what
makes document-chunked sparsity exploitable.** This is consistent with round 1's finding that the
BLOCK_N pow2 clamp only moved the number 1.6% — and it retires the tile lever in the other
direction too, since a larger `kv_block_size` coarsens kv sparsity the same way.

`arm04_q128` also shows what "correct" looks like when numerics shift: 22/24 generations matched and
f1 moved 0.750 → 0.756 (well inside ±0.088). That is greedy decoding diverging on floating-point
reduction order, not corruption — corruption shows up as `parse_rate` 0 and "!!!!".

**`kv_block_size` ∈ {32, 64, 128} never ran**: the guardrail already in `vllm_chunked_patch.py`
rejects any `kv_block_size` that does not divide the 528-token KV page. See §5 for why I left that
guardrail standing rather than measuring through it.

### 2b. Concurrency — the real lever, but it saturates fast

`debug/chunked_eval_speedup/conc_sweep.sbatch` (job 3438001), `eval_size=48`, `kv_block=16`.
`max_num_batched_tokens` is lowered in step with `max_num_seqs` to hold the prefill mask temporary
near the ~190 MB the production config already pays (see the script header for the budget).

| arm | max_num_seqs | bt | gen_s | vs base | steps | set_f1 | parse_rate | gen_match |
|---|---|---|---|---|---|---|---|---|
| `c00_base` | 8 | 2048 | **289.1** | 1.00x | 3129 | 0.7361 | 1.000 | 48/48 |
| `c01_seqs16` | 16 | 1024 | 254.6 | 1.14x | 1687 | 0.7569 | 1.000 | 41/48 |
| `c02_seqs32` | 32 | 512 | **242.4** | **1.19x** | 1331 | 0.7500 | 1.000 | 36/48 |
| `c03_seqs48` | 48 | 512 | 249.8 | 1.16x | 1324 | 0.7500 | 1.000 | 35/48 |
| `c05_base_bt512` *(control)* | 8 | 512 | 266.0 | 1.09x | 3293 | 0.7639 | 1.000 | 33/48 |

**The optimum is 32 and it turns over immediately after**: `c03_seqs48` removes essentially no
further steps (1331 → 1324) and is *slower* than `c02_seqs32`.

**And the control splits the win in half.** `c05_base_bt512` holds concurrency at the production 8
and only drops `max_num_batched_tokens` 2048 → 512 — worth **1.09x on its own**, with *more* steps
(3293 vs 3129). So the 1.19x at `c02` decomposes as:

| lever | factor |
|---|---|
| `max_num_batched_tokens` 2048 → 512 | 1.09x |
| `max_num_seqs` 8 → 32 (on top of that) | 1.10x |
| **combined** | **1.19x** |

Smaller prefill chunks help because a prefill step's mask temporary is
`max_num_batched_tokens x total_cache_tokens` — 197 MB at bt=2048 vs 49 MB at bt=512 — and the
allocator, not the arithmetic, is what that costs (the total number of mask elements evaluated is
identical either way, since 4x more prefill steps each do 4x less).

Widening decode from 8 to 32 resident sequences removes **57% of the forward steps** (3129 → 1331)
and buys only **1.19x**. Per-step cost went the other way: 92.4 ms → 182.1 ms.

**This is where the round-1 / round-2a KV-allocation hypothesis is actually true — it just isn't a
free win.** The fallback mask build is `O(num_actual_tokens x total_cache_tokens)`, and holding N
sequences requires a cache of N·L tokens, so a *decode* step costs `O(N x N·L)` — quadratic in
concurrency. Every step you remove makes the remaining steps more expensive, at almost exactly the
rate that cancels the gain. Fitting `cost = a + bN` to c00/c01 gives a ≈ 34 ms fixed + 7.3 ms per
resident sequence, whose asymptote (`N → ∞`) is only ~1.65x — and the true curve is worse than that
linear fit because the mask term is quadratic. c02 landing at 1.19x is consistent with the
quadratic, not the linear, model.

So: concurrency is the largest single lever measured, and it is worth ~1.2x, not the ~4x that "57%
fewer steps" suggests.

`gen_match` degrades (48/48 → 41/48 → 36/48) purely because batch composition changes floating-point
reduction order in the kernels; `set_f1` stays flat at 0.736 / 0.757 / 0.750, all within the ±0.088
binomial SE at `eval_size=48`. ⚠ These per-arm f1 values are all `eval_size=48` only and are used
here as corruption detectors, not as measurements — the real correctness gate is §4.

### 2c. CUDA graphs — CRASHES. `enforce_eager=True` is load-bearing, not vestigial.

`arm08_cudagraph` (`enforce_eager=False`, vLLM's default cudagraph mode) captures graphs fine and
then dies on first replay:

```
Capturing CUDA graphs (mixed prefill-decode, PIECEWISE): 5/5
Capturing CUDA graphs (decode, FULL): 4/4
[rank0]: torch.AcceleratorError: CUDA error: device-side assert triggered
```

A FULL decode graph bakes in the tensors it captured, but the chunked patch allocates a **brand new
`block_mask` on every step** through `create_block_mask`, so the replay walks stale pointers. vLLM's
FlexAttention cudagraph support (`AttentionCGSupport.ALWAYS`) is built on the direct-build path's
persistent buffers (`persistent_kv_indices`, and note the "+1 sentinel column" comment at
`flex_attention.py:891` about the kernel prefetching one past the last valid index) — machinery
chunked mode never touches because it is permanently on the fallback path.

**The answer to "does the monkey-patch actually require `enforce_eager=True`?" is yes.** All three
modes were measured (`combo_sweep.sbatch`, job 3438049, `eval_size=24`, baseline 111.4s):

| arm | `cudagraph_mode` | result |
|---|---|---|
| `arm08_cudagraph` | vLLM default (FULL + PIECEWISE) | `CUDA error: device-side assert triggered` |
| `d01_piecewise` | `PIECEWISE` | `CUDA error: an illegal memory access was encountered` |
| `d02_compile` | `NONE` (torch.compile, no graphs) | runs — **118.2s, 6% SLOWER**, and load 14s → 36s |

PIECEWISE was the hope, since splitting at the attention op should leave the BlockMask outside every
capture. It does not survive either. And with graphs off entirely, torch.compile on its own is a
small net loss. **The whole CUDA-graph / compilation lever is dead for this path**, and the
`enforce_eager=True` in `run_vllm_eval.py` should be documented as load-bearing rather than left
looking like a leftover. The failures are crashes, not wrong numbers, which is the good outcome.

### 2d. Chunk-id cache — free, exact, and small

`CHUNK_CACHE_IDS=1` caches each request's chunk-id row and extends it, falling back to a full
rescan whenever the newly-generated tokens contain a doc marker. Measured twice independently:

| arm | gen_s | vs base | set_f1 | parse_rate | gen_match |
|---|---|---|---|---|---|
| `arm07_cache` (job 3437991) | 109.6 | 1.016x | 0.7500 | 1.000 | **24/24** |
| `d03_cache` (job 3438049) | 109.5 | 1.017x | 0.7500 | 1.000 | **24/24** |

**Byte-identical generations**, as the construction requires. Correctness is proven, not sampled:
`debug/chunked_eval_speedup/test_chunk_ids_cache_parity.py` checks the cached and uncached paths
elementwise over a prefill-then-decode trajectory, including the two cases the argument turns on —
a model that *emits* `<|doc_start|>` / `<|doc_end|>` mid-generation (must force a rescan) and a slot
reused by a different request id. It also measures the stage in isolation: 4.65 ms → 0.63 ms per
decode step at 8 resident 8875-token sequences.

Worth 1.6-1.7%, exactly as the §1 attribution predicts (`build_chunk_ids` is 2.5% of runtime and the
cache removes ~75% of it). Take it because it is free and provably safe, not because it matters.

### 2e. The two levers interact — and eval_size decides whether concurrency pays at all

`combo_sweep.sbatch` (job 3438049), `eval_size=24`, baseline 111.4s (measured with the node
otherwise idle; the 111.3s in §2a had a co-tenant, so contention was never a confound):

| arm | config | gen_s | vs base | set_f1 | parse_rate | gen_match |
|---|---|---|---|---|---|---|
| `arm00_base` | production | **111.4** | 1.00x | 0.7500 | 1.000 | 24/24 |
| `d02_compile` | `cudagraph_mode=NONE` | 118.2 | 0.94x | 0.7556 | 1.000 | 23/24 |
| `d03_cache` | `CHUNK_CACHE_IDS=1` | 109.5 | 1.02x | 0.7500 | 1.000 | 24/24 |
| `d04_seqs32` | seqs 32, bt 512 | 166.7 | **0.67x** | 0.7778 | 1.000 | 16/24 |
| `d05_cache_seqs32` | cache + seqs 32, bt 512 | 110.9 | 1.00x | 0.7778 | 1.000 | 16/24 |
| `d06_cache_seqs64` | cache + seqs 64, bt 256 | 199.2 | 0.56x | 0.7694 | 1.000 | 17/24 |

Two things here matter more than any single number.

**(i) The chunk-id cache is worth 1.6% at 8 resident sequences and 1.50x at 32.** `d04` → `d05` is
166.7s → 110.9s from nothing but `CHUNK_CACHE_IDS=1`. `_build_chunk_ids_for_batch` costs
`O(num_reqs x seq_len)` per step (4.65 ms at 8 reqs, 18.5 ms at 32 — see the microbenchmark in §1),
so it grows exactly as fast as concurrency does. **The cache is not a rounding error, it is the
thing that makes the concurrency knob usable at all** — without it, raising `max_num_seqs` pays for
its own step savings twice over.

**(ii) `max_num_seqs` must be sized against `eval_size`, and 24 examples is the wrong yardstick.**
The identical seqs=32 config is **1.19x faster at eval_size=48** and **1.5x slower at eval_size=24**.
The mechanism is not subtle: when `eval_size <= max_num_seqs` every request is resident from the
first step, so the cache allocation (and therefore the `O(N x N.L)` mask build) is paid in full while
there is no queue left to drain and no steps to save. `d06`'s seqs=64 is the same failure, worse.

This is the trap that would have made a small-sample sweep recommend the wrong production config,
and it is why §4 re-runs the candidates at the full `eval_size=500`.

## 3. Full-rung validation at the production eval_size (the only numbers to trust)

`debug/chunked_eval_speedup/fullrung_validate.sbatch`. Each arm consumes **the exact prefills pack
the known-good production run consumed** (`ctc_suite_vllm_results_chunked/contradiction_iid/
prefills_8192.json`, 500 rows), so wall clock and f1 are both directly comparable.

Gate: `set_f1` within the 500-example binomial SE (0.018) of 0.8039, **and** `parse_rate == 1.000`.

| arm | config | gen_s | speedup | steps | set_f1 | Δ | parse_rate | verdict |
|---|---|---|---|---|---|---|---|---|
| production baseline | seqs 8, bt 2048 | 2066.1 | 1.00x | 32573 | 0.8039 | — | 1.000 | reference |
| **`cand_cache_seqs32_bt2048`** | **cache + seqs 32 + bt 2048** | **1163.9** | **1.78x** | 8355 | **0.7999** | −0.0040 | **1.000** | **REPRODUCES** |
| `cand_cache_seqs32` | cache + seqs 32 + bt 512 | 1616.6 | 1.28x | 9215 | 0.8000 | −0.0039 | 1.000 | reproduces |
| `cand_cache_seqs64` | cache + seqs 64 + bt 512 | 1755.3 | 1.18x | 9215 | 0.8000 | −0.0039 | 1.000 | reproduces |
| `cand_cache_bt512` | cache + seqs 8 + bt 512 | 2232.4 | **0.93x** | 33396 | 0.8035 | −0.0004 | 1.000 | reproduces, but SLOWER |

Three things to take from this table.

**The winner is validated: 1.78x, f1 0.7999 vs 0.8039 (Δ −0.0040, well inside the ±0.018 SE),
parse_rate 1.000.** Forward steps drop 3.9x, 32573 → 8355.

**`max_num_batched_tokens=512` is a consistent loss at production scale.** Same concurrency, only
the prefill budget differs: 1163.9s at bt=2048 vs 1616.6s at bt=512 — holding it at the production
2048 is worth 1.39x on its own. And at production concurrency it is a 7% loss outright
(`cand_cache_bt512`, 0.93x). **Leave it alone.**

**Concurrency still turns over at 32**: `cand_cache_seqs64` is slower than `cand_cache_seqs32`,
reproducing at eval_size 500 the same turnover `c03_seqs48` showed at 48.

**And the bottom row is why nothing smaller than a full rung should be believed.** The
`max_num_batched_tokens=512` change measured **1.09x faster at eval_size=48** (`c05_base_bt512`) and
is **0.93x — a loss — at eval_size=500**, producing *more* steps than baseline (33396 vs 32573).
Combined with `d04_seqs32` flipping sign between eval_size 24 and 48 (§2e), two of the three knobs in
this study have an eval_size-dependent sign. Tuning on a 24- or 48-example sample would have shipped
`bt=512` and left 1.39x on the floor.

## 4. Recommended production config

**Nothing in this document has been applied.** `run_vllm_eval.py` and `gpu_eval_task_chunked.sh`
still carry their historical defaults; every knob below is an environment variable that defaults to
today's behavior, so setting none of them reproduces the current numbers exactly.

Set these in the chunked launcher (`debug/ctc_vllm_validation/node_local/gpu_eval_task_chunked.sh`):

```sh
export CHUNK_CACHE_IDS=1      # incremental chunk-id cache; provably exact (§2d)
export CHUNK_MAX_NUM_SEQS=32  # was 8
export CHUNK_SEQ_HEADROOM=34  # KV pages must actually hold 32 resident sequences
# and do NOT set CHUNK_MAX_BATCHED_TOKENS -- leave it at its 2048 default.
```

**Expected: 1.78x** (8k contradiction, 500 examples: 2066.1s → 1163.9s), with `set_f1` 0.8039 →
0.7999 and `parse_rate` 1.000. Measured end-to-end on the production prefills pack, not extrapolated.
Two environment variables; no code path changes.

Caveats a reader should not have to rediscover:

* **Do not "help" by also lowering `max_num_batched_tokens`.** It is the intuitive companion change
  (it bounds the prefill mask temporary) and it costs 1.39x: same concurrency, 1163.9s at bt=2048 vs
  1616.6s at bt=512. At production concurrency it is a loss outright (0.93x).
* **`CHUNK_MAX_NUM_SEQS` must be re-checked per rung, not copied.** Its benefit comes from having a
  queue to drain; its cost (`O(N x N.L)` mask build) grows with rung length `L`. 32 is validated at
  **8k with eval_size 500**, and 64 is already worse there (1.18x). At 32k, `L` is 4x larger, so the
  same 32 costs 4x more per step — this is the setting most likely to need lowering, and it has not
  been measured above 8k.
* Do not raise `max_num_seqs` above `eval_size`. It inverts (§2e).
* `CHUNK_SEQ_HEADROOM` must move with `CHUNK_MAX_NUM_SEQS` — it is what sizes the KV allocation to
  actually hold that many sequences.

### What NOT to change (each measured, each a loss)

| change | measured result |
|---|---|
| `flex_attn_q_block_size=128` | 0.41x — 2.5x slower (§2a) |
| `flex_attn_kv_block_size` 32/64/128 | blocked by the divisibility guardrail; not measured (§5) |
| `enforce_eager=False` (default cudagraphs) | `device-side assert triggered` |
| `enforce_eager=False, cudagraph_mode=PIECEWISE` | `illegal memory access` |
| `enforce_eager=False, cudagraph_mode=NONE` | 0.94x, plus load 14s → 36s |
| shrinking the KV allocation (round 2a) | 0.44x at headroom 2 — refuted |
| `max_num_batched_tokens=512` | 0.93x at production concurrency; costs 1.39x at seqs=32 (§3) |

## 5. The correctness landmine, and what I could not make safe

### 5a. `_build_block_mask_direct()` — now raises instead of falling through

`vllm_chunked_patch.py` contained:

```python
if metadata.direct_build and metadata.causal:
    metadata.block_mask = metadata._build_block_mask_direct()
else:
    ...                       # the fallback that DOES apply the chunked mask_mod
```

`_build_block_mask_direct()` builds the BlockMask from the page table and causal structure alone —
**it never evaluates `mask_mod`**. Taking that branch silently discards the chunked mask and yields
dense-causal numbers wearing a "chunked" label: the worst possible failure, because it reads as a
modelling result rather than a bug.

**It is dead code today, and that is now confirmed rather than assumed.** The production 8k run's
`patch_debug` reads `{"direct": 0, "fallback": 32573}` — every builder call took the fallback. vLLM
sets `direct_build = False` whenever `kv_block_size != block_size` (`flex_attention.py:882`), and the
Qwen3.5 GDN-hybrid's attention page is 528 while `kv_block_size` must be a power of two, so they can
never be equal here.

**I changed the branch to `raise RuntimeError` with an explanatory message** rather than leave a
silent fallthrough armed for whoever next changes a block size. This is the one behavioral edit in
this round; it converts an unreachable-but-catastrophic path into an unreachable-but-loud one, and
it cannot affect any run that does not already take the branch. (Belt and braces: vLLM's own
`_build_block_mask_direct` would also raise on our sizes, since it asserts
`kv_block_size // block_size == 1`. The patch-level raise is the one that carries the *why*.)

### 5b. A STALE SECOND COPY of the patch that WOULD drop the chunked mask — unfixed, flagged

There are two copies of this module:

* `src/corpus_reasoning/lib/vllm_chunked_patch.py` — **the live one** (`run_vllm_eval.py` imports
  `from corpus_reasoning.lib import vllm_chunked_patch`).
* `src/scripts/ctc_eval/lib/vllm_chunked_patch.py` — **stale, and dangerous.**

The stale copy's fallback branch is:

```python
metadata.block_mask = metadata.build_block_mask()
```

`FlexAttentionMetadata.build_block_mask()` recomputes `self.get_mask_mod()` internally — the
**default causal** mask_mod — so it silently discards the chunked `mask_mod` the patch just
installed, on the branch that chunked mode *always* takes. The live copy was fixed for exactly this
(its comment marks it "⚠"); the `ctc_eval` copy never was, and it also lacks `get_debug_state()`, so
a run through it could not even report `direct`/`fallback` counts.

**I did not touch it** — it is outside this task's scope and deleting a module could break an import
I have not traced. But anything that starts importing `scripts.ctc_eval.lib.vllm_chunked_patch` will
produce dense numbers labelled chunked, with no error. It should be deleted or reduced to a re-export.

### 5c. `kv_block_size` — deliberately NOT measured

The tile sweep's `kv_block_size` ∈ {32, 64, 128} arms never ran: the guardrail already in the patch
rejects any `kv_block_size` that does not divide the 528-token KV page, citing the historical
"!!!!" corruption.

I believe that guardrail is over-broad on the fallback path — there, `kv_block_size` is only the
BlockMask's granularity over flat physical KV indices, and `flex_attention` re-applies `mask_mod`
elementwise inside every partial block, so any grouping should be exact; the constraint is real only
for `_build_block_mask_direct`, which maps pages to blocks 1:1. **I did not act on that belief.**
Reasons, in order:

1. **The measured payoff is small and probably negative.** `arm04_q128` showed that widening the
   *other* block dimension cost 2.5x, because block size is the resolution of the chunk sparsity, not
   just a tile size. A larger `kv_block_size` coarsens kv sparsity the same way. §1 also caps the
   prize: all GPU work is 32% of runtime, and the attention kernel is only part of that.
2. **The failure mode is the silent-corruption class**, and I cannot explain the historical
   "!!!!" observation under my reading — which means my reading is incomplete.

I added an opt-in escape hatch, `CHUNK_ALLOW_KVBLOCK_MISMATCH=1`, which downgrades the guardrail to a
loud warning, so the question is one env var away from being settled. **It has never been run.**
Anyone who does should gate on `parse_rate` and a full-rung f1, exactly as §3 does.

## 6. Loose ends

* All four full-rung candidates completed and are in the §3 table; none is outstanding.
* **Everything here is 8k contradiction only.** The `max_num_seqs` optimum is rung-dependent by
  construction (its mask cost scales with `L`), so 16k and 32k need their own check before the
  setting is applied ladder-wide.
* The remaining ~50% of runtime is unattributed per-step CPU outside the metadata builder — vLLM's
  scheduler, the eager forward launch, sampling, detokenisation. That is the only place a large
  further win could come from, and CUDA graphs (the standard tool for it) are ruled out by §2c. A
  serious attempt would mean giving FlexAttention a persistent-buffer path for a custom `mask_mod`,
  i.e. upstream work in vLLM, not a config change.

## 7. Artifacts

| path | what |
|---|---|
| `debug/chunked_eval_speedup/kvcache_sweep.sbatch` | round 2a, KV-allocation hypothesis (refuted) |
| `debug/chunked_eval_speedup/tile_sweep.sbatch` | tile sizes + cudagraphs + cache, `eval_size=24` |
| `debug/chunked_eval_speedup/conc_sweep.sbatch` | `max_num_seqs` / `max_num_batched_tokens`, `eval_size=48` |
| `debug/chunked_eval_speedup/combo_sweep.sbatch` | cudagraph modes + lever interactions, `eval_size=24` |
| `debug/chunked_eval_speedup/fullrung_validate.sbatch` | the `eval_size=500` gate (parameterised by `ARM` + `CHUNK_*`) |
| `debug/chunked_eval_speedup/bench_chunk_ids_r2.py` | CPU microbenchmark that exposed the sync artifact |
| `debug/chunked_eval_speedup/test_chunk_ids_cache_parity.py` | elementwise parity proof for `CHUNK_CACHE_IDS` |

Job logs on cubbins: `/data/prasann/joblogs/chunk_{kv_sweep,tile_sweep,conc_sweep,combo,fullrung}_*.log`.

New env knobs (all default to historical behavior): `CHUNK_CACHE_IDS`, `CHUNK_KV_BLOCK`,
`CHUNK_Q_BLOCK`, `CHUNK_ENFORCE_EAGER`, `CHUNK_CUDAGRAPH_MODE`, `CHUNK_PROFILE_SPLIT`,
`CHUNK_ALLOW_KVBLOCK_MISMATCH`, plus round 2a's `CHUNK_SEQ_HEADROOM`, `CHUNK_MAX_BATCHED_TOKENS`,
`CHUNK_MAX_NUM_SEQS`.

---

# Round 3 — "chunked should be at least as fast as dense"

Round 2 optimized within the existing representation and got 1.78x. The target was then restated:
chunked expresses a block-diagonal pattern, so it should **beat** dense, not trail it. This section
answers that, and it starts by correcting two premises — one of mine, one of the reframing's.

## 8. Correction: how `mask_mod` actually reaches the kernel (§5b restated)

My §5b warning said the stale `src/scripts/ctc_eval/lib/vllm_chunked_patch.py` copy "recomputes the
default causal mask_mod". That phrasing was imprecise and the review correctly pushed back: **both**
copies do set `metadata.mask_mod` to the chunked variant (stale line 382, live line 491). Here is
the actual mechanism, traced end to end.

1. `FlexAttentionImpl.forward` (`flex_attention.py:1360-1385`) calls
   `flex_attention_compiled(query, key, value, attn_metadata.transformed_score_mod,
   attn_metadata.block_mask, ...)`. **It never passes `attn_metadata.mask_mod`.**
2. A mask reaches the kernel only as a field *inside* the BlockMask:
   `BlockMask.as_tuple()` (`torch/nn/attention/flex_attention.py`) returns
   `(..., q_indices, full_q_num_blocks, full_q_indices, *block_size, self.mask_mod)` — the trailing
   element is the mask_mod, and it is what the kernel applies inside partial blocks.
3. So the mask that runs is **`block_mask.mask_mod`** — i.e. whatever function was handed to
   `create_block_mask`, not whatever was assigned to `metadata.mask_mod`.

That settles both questions:

* **The live path is correct.** `corpus_reasoning/lib` calls
  `create_block_mask_compiled(metadata.mask_mod, ...)` explicitly, so the chunked rule is baked into
  `block_mask.mask_mod` and does reach the kernel. Our published chunked numbers are genuinely
  chunked. Nothing needs re-running.
* **The stale copy is still broken, for the reason I gave.** It calls `metadata.build_block_mask()`,
  whose first line is `mask_mod = self.get_mask_mod()` (`flex_attention.py:625`) — rebuilding from
  `get_paged_mask_mod()` and ignoring the assignment made moments earlier. The resulting BlockMask
  carries the **default causal** mask_mod, so on that path the chunked rule is silently dropped.
  Assigning `metadata.mask_mod` is inert there because nothing downstream reads it.

The review's alternative reading — "mask_mod applies at kernel level while block_mask only controls
block skipping" — is **not** how it works: there is no separate kernel-level mask channel, and
block_mask is not merely a skipping hint. The two are the same object. The warning stands as
written in §5b; only my wording of *why* needed fixing.

## 9. Where the time actually goes: 93.7% of steps are DECODE, not prefill

The varlen reframing rests on prefill being the expensive part ("input ~2100 tok/s dominates; output
is ~32 tok/s on a ~35-token answer"). The production 8k run's own numbers say otherwise. From
`prefills_8192.json` (500 rows, 4,208,248 prompt tokens, mean 8416) and `patch_debug.calls = 32573`
at `max_num_batched_tokens=2048`, `max_num_seqs=8`:

| | steps | share |
|---|---|---|
| prefill | 4,208,248 / 2048 ≈ **2,055** | **6.3%** |
| decode | 32573 − 2055 ≈ **30,518** | **93.7%** |

Back out the generation length: `30518 x 8 / 500` = **~488 generated tokens per example**, against
`max_new_tokens=512`.

**The answers are ~34 characters.** Median/mean/max response length over all 500 is 34/34/37 chars —
e.g. `[[27, 114], [49, 97], [73, 100]]`, about 18 tokens. The model generates ~488 tokens and
`truncate_like_native` throws away ~470 of them: contradiction uses `stop_rule=eos`, the model never
emits EOS, so every request runs to the 512 cap and is cut post-hoc at the first newline after the
answer completes.

Two consequences, and they reshape the whole problem:

**(a) Varlen prefill addresses 6.3% of the steps.** Making prefill infinitely fast is bounded above
by roughly 1.07x on step count. Whatever the block-diagonal cost argument says about FLOPs, it is
aimed at the part of this workload that barely runs.

**(b) Chunking cannot help decode *at all*, by construction.** Generated tokens are FREE, and the
rule `allowed = causal & not_pad & (context_ok | q_free | kv_free)` gives a FREE query
`allowed = causal`. A decode query attends to the entire context in both arms. Chunked decode has
exactly the same FLOPs as dense decode — it can match dense, never beat it.

So "chunked should be much faster than dense" is true of **prefill** and false of **decode**, and
this eval is 93.7% decode. The 17.8x deficit is not the mask being expensive; it is that chunked is
capped at `max_num_seqs=8` while dense runs at vLLM's default 1024, which costs chunked ~62x more
decode steps (30,518 vs ~490).

## 10. Varlen segments for chunked prefill — the cost model is right, the target is wrong

The proposed decomposition is correct as arithmetic. The pure chunked rule splits into

1. **context -> context**: block-diagonal, `O(sum_i L_i^2)` — natively expressible as `cu_seqlens`.
2. **FREE -> everything**: a thin dense strip, `O(|FREE| * N)`.
3. context -> leading-FREE (instruction prefix): a second thin strip; context -> *trailing* FREE is
   causally excluded, as noted.

versus dense `O(N^2)`. At the 32k rung (n=762 docs, ~9 tokens each) term 1 is ~3 orders of magnitude
below dense. None of that is in dispute.

**It cannot fix this eval, because it optimizes prefill and this eval is 93.7% decode (§9).** Even a
free prefill leaves the decode path — which chunking provably cannot improve, since every generated
token is FREE and attends to the whole context — untouched. The upper bound on the whole idea, at
the current generation length, is a few percent.

### What it would additionally cost, if someone still wants it later

A repo survey (see §12) turned up two facts that matter:

* **The recombine pattern cited as precedent does not exist.** The sentence in
  `landmark_prefill_sparse.py:27-31` about "sort queries by assigned block, varlen-attend per block,
  recombine with online softmax" is describing **MoBA, a published alternative being contrasted
  with this module's design** — not this module's implementation. What the file actually does is a
  single fused Triton kernel with one program per `(query_block, batch*head)` and in-kernel running
  online-softmax accumulation over a data-dependent candidate list. There is no per-group varlen
  call and no external merge.
* **There is no LSE-merge helper anywhere in the repo.** `merge_attn_states`, `return_softmax_lse`
  and `softmax_lse` have zero hits. Every online-softmax merge in the codebase (landmark_kernel,
  landmark_compressive, landmark_fast, landmark_sparse_kernel, landmark_prefill_sparse) is either
  single-pass in-kernel accumulation or an LSE tensor saved solely to drive a custom Triton
  *backward* pass. The "run attention twice, merge two `(O, LSE)` pairs" primitive would be written
  from scratch.

What *does* exist is single-pass block-diagonal varlen: `cu_doc_lens` ->
`dispatch_flash_attn` -> `flash_attn_varlen_func` (`flash_attn_api.py:104`), used by the landmark and
recurrent paths. That handles term 1 only. It cannot express terms 2-3, which is exactly why
`document_chunked.py:399-416` raises `NotImplementedError` on `cu_doc_lens` today rather than using
the varlen path that sits right next to it.

So the work is: a from-scratch multi-pass + LSE-merge attention primitive, implemented inside
vLLM's paged-KV attention backend (where the "documents" are scattered across pages and cu_seqlens
would have to be rebuilt per step from the page table), to accelerate 6.3% of the steps. **Not
recommended at the current generation length.** It becomes worth revisiting only if §11's early-stop
change lands, which inverts the prefill:decode ratio.

## 11. What actually worked: the FREE-query fast path, and early stopping

### 11a. FREE-query fast path — skip the chunked mask when it provably cannot bind

Since a FREE query makes the chunked rule collapse to plain causal (§9b), any step whose query
tokens are **all** FREE can reuse the paged-causal BlockMask vLLM already built in `build()`
(`flex_attention.py:1141`) instead of rebuilding a chunked one. `CHUNK_FREE_QUERY_FASTPATH=1` checks
that condition per step, against the actual chunk ids of the actual query positions, and returns
vLLM's metadata untouched when it holds.

This is not an approximation. `debug/chunked_eval_speedup/test_free_query_equivalence.py` verifies
elementwise over 300 randomized document layouts that the chunked and causal masks agree on every
FREE query row (21,536 rows), that they genuinely *disagree* on context rows (so the test is not
vacuous), and that the gate fires on an all-FREE decode step, refuses a step containing a context
query, and refuses a full-sequence prefill.

The payoff is that it removes the `O(num_actual_tokens x total_cache_tokens)` build from the steps
that were capping concurrency — so `max_num_seqs` can finally be raised:

| arm | config | gen_s | speedup | steps | freepath hits | set_f1 | parse | verdict |
|---|---|---|---|---|---|---|---|---|
| baseline | seqs 8 | 2066.1 | 1.00x | 32573 | — | 0.8039 | 1.000 | reference |
| `fp_only` | fp + cache, **seqs 8** | 1789.5 | 1.15x | 32573 | 30358 | **0.8039** | 1.000 | **Δ = +0.0000** |
| round-2 best | cache + seqs 32 | 1163.9 | 1.78x | 8355 | — | 0.7999 | 1.000 | reproduces |
| `fp_seqs256` | fp + cache + seqs 256 | 1069.1 | 1.93x | 2676 | 511 | 0.8001 | 1.000 | reproduces |
| **`fp_seqs64`** | **fp + cache + seqs 64** | **861.4** | **2.40x** | 4385 | 2231 | **0.8008** | **1.000** | **REPRODUCES** |

**`fp_only` is the correctness proof.** Holding concurrency at the production 8 so that *nothing*
changes except whether the fast path is taken, it fires on **30,358 of 32,573 steps (93.2%** —
independently confirming the 93.7% decode share derived in §9) and returns
**set_f1 0.8039, Δ = +0.0000 against the known-good number**, at identical step count. A change that
touches 93% of steps and moves the metric by zero in the fourth decimal is doing exactly what the
equivalence proof says it does. (It is only worth 1.15x on its own, because at `max_num_seqs=8` the
bottleneck was never the mask build — it was the step count. The fast path's value is that it
*unlocks* the concurrency knob below.)

Round 2 found concurrency saturating at 32 (64 was *worse*, 1.18x). With the fast path, 64 is the
optimum and worth 2.40x — the ceiling moved because the thing that made extra concurrency expensive
was the per-decode-step mask build, not memory. Beyond 64 it turns over again for a different
reason: `max_num_seqs=256` needs a far larger KV cache, and the **prefill** steps (which still take
the chunked path, `applied=2165`) pay `O(2048 x total_cache_tokens)` for it.

### 11b. Early stopping — the biggest single win, and nearly free

§9 showed the model generates ~488 tokens per example and `truncate_like_native` discards ~470 of
them. Since its completion test for `cot_mode="none"` is `"]]" in ans`, stopping the sampler at
`]]` cuts generation exactly where the answer is already complete. `CHUNK_EXTRA_STOP=']]'`:

| arm | gen_s | vs own baseline | steps | set_f1 | Δ vs known-good | parse | verdict |
|---|---|---|---|---|---|---|---|
| `fp_stop` = fp + cache + seqs 64 + `]]` | **311.9** | **6.62x** | 2089 | **0.8032** | **−0.0007** | 1.000 | **REPRODUCES** |
| `dense_stop` = dense + `]]` | **77.8** | **1.50x** | — | **0.9764** | **+0.0004** | 1.000 | **REPRODUCES** |

Δ of −0.0007 and +0.0004 are the tightest agreements in this entire study — much tighter than any
config-only arm — which is what you expect from a change that only stops generating tokens that were
being thrown away anyway.

Note the asymmetry: early stopping is worth **6.62x/2.40x = 2.76x** on top of the chunked fast-path
config, but only **1.50x** for dense. Dense already batched all 500 sequences at once, so its decode
was ~490 cheap steps and it was prefill-bound already; chunked was drowning in decode.

### 11c. Where that leaves "chunked <= dense"

**Not achieved. Chunked is 4.0x slower than dense when both run their best validated config**
(311.9s vs 77.8s). The 17.8x is down to 4.0x, and the residual has changed character completely:
`fp_stop` runs 2089 steps of which ~2055 are prefill — **the eval is now ~98% prefill.**

Which means §10's verdict inverts. Varlen/block-diagonal prefill was the wrong lever against a
93.7%-decode workload; against a 98%-prefill workload it is precisely the right one, and it is now
the *only* remaining lever large enough to close a 4.0x gap. The block-diagonal term is orders of
magnitude below dense at these document counts, so chunked prefill *should* beat dense prefill
outright — it currently loses because it is expressed as a custom mask over a dense layout, which
costs both the `O(bt x total_cache_tokens)` BlockMask build and the unoptimized FlexAttention
kernel.

**Recommended next step, in order:** land 11a + 11b first (they are cheap, validated, and they are
what makes prefill the bottleneck worth attacking), then reassess varlen against the new profile.

## 12. Revised recommendation (supersedes §4)

Still nothing applied; all knobs default to today's behavior.

**Tier 1 — config only, no generation change. 2.40x.**

```sh
export CHUNK_FREE_QUERY_FASTPATH=1  # skip the chunked mask on all-FREE-query steps (§11a)
export CHUNK_CACHE_IDS=1            # incremental chunk-id cache (§2d)
export CHUNK_MAX_NUM_SEQS=64        # was 8; the fast path is what makes this pay
export CHUNK_SEQ_HEADROOM=66
```
8k contradiction, eval_size 500: 2066.1s → 861.4s, set_f1 0.8039 → 0.8008, parse_rate 1.000.

**Tier 2 — adds an early stop. 6.62x, and it applies to the dense arm too.**

```sh
export CHUNK_EXTRA_STOP=']]'        # contradiction / cot_mode="none" ONLY -- see caveat
```
Chunked 861.4s → 311.9s (set_f1 0.8032, Δ −0.0007). Dense 116.4s → 77.8s (set_f1 0.9764, Δ +0.0004).

⚠ **The stop string is task-specific and must not be applied blindly.** `]]` is correct only where
`truncate_like_native`'s completion test is `"]]" in ans`, i.e. contradiction with
`cot_mode="none"`. For `cot_mode != "none"` the test is `"contradicting pairs:" in txt`, and other
tasks route through `truncate_generic` with different rules entirely. Each task needs its stop
string derived from its own truncation rule and validated at eval_size 500. Two known hazards: a
`[]` (empty) answer contains no `]]` and will still run to the cap (harmless, just not accelerated),
and a model that emitted `]]` inside a `<think>` block would be cut early — not observed here (all
500 responses are clean 29-37 char answers) but it is the thing to check per task.

**What NOT to do (measured, this round):**

| change | result |
|---|---|
| `max_num_seqs=256` with fast path | 1.93x — worse than 64 (prefill pays for the bigger cache) |
| varlen/cu_seqlens prefill, *before* early stopping | targets 6.3% of steps; see §10 |

**Open, and now the only lever big enough to matter:** chunked is still **4.0x slower than dense**
(311.9s vs 77.8s) and the workload is now ~98% prefill. That gap is the dense-layout custom-mask
representation, and closing it is the varlen work of §10 — which should be re-scoped against the
post-early-stop profile, not the one it was originally proposed against.

## 13. Round-3 artifacts

| path | what |
|---|---|
| `debug/chunked_eval_speedup/test_free_query_equivalence.py` | elementwise proof that chunked ≡ causal on FREE query rows, + gate tests |
| `debug/chunked_eval_speedup/fullrung_validate.sbatch` | eval_size=500 gate; now takes `MODE=full` for the dense arm |

New env knobs (all default to today's behavior): `CHUNK_FREE_QUERY_FASTPATH`, `CHUNK_EXTRA_STOP`.

Job logs on cubbins: `chunk_fullrung_3438392` (fp_only), `_3438393` (fp_seqs64), `_3438394`
(fp_seqs256), `_3438400` (fp_stop), `_3438456` (dense_stop).

One behavioral edit was made to the patch module across rounds 2-3: the `direct_build` branch now
raises instead of silently dropping the chunked mask (§5a). Everything else is env-gated and off by
default; `CHUNK_*`-free imports were re-verified to reproduce the original code path.

---

# Round 5 — the simple config, the varlen rewrite, and a confound that rewrites round A

Rounds 5a-5f (2026-08-13) restarted from scratch at SMALL scale (smoke model
`ctc-smoke-contra-3ep`, 0.5B GDN hybrid; contradiction rung 2560; eval_size=100 ⚠ perf bench —
scores gate corruption only) and ended with two production-validated results and one
methodological correction that invalidates part of round 5a.

## 14. The small-scale ladder (jobs 3443501 / 3443561)

Dense + stop = 2.7s. The historical chunked config = **286.6s (106x)**. Additive findings:

| change | gen_s | takeaway |
|---|---|---|
| historical chunked | 286.6 | the bug |
| + `]]` early stop | 21.2 | **13.5x from the stop alone** — decode waste dominates everywhere |
| + one-shot prefill (bt >= prompt) + seqs16 | **15.4** | one BlockMask build per REQUEST |
| round-3 tier-2 (fp+cache+seqs) | 87.0 | *phantom* — see §17 |

Two structural points survived contention-controlled re-measurement (§17):

* **One-shot prefill** (`CHUNK_MAX_BATCHED_TOKENS >= longest prompt) turns ceil(len/2048) chunked
  BlockMask builds per prompt into exactly one, and it needs moderate concurrency to pay
  (b4 at seqs8 was 47.6s: with nothing else resident, a one-shot step serializes).
* fp / cache / seqs16 measured INDIVIDUALLY are all within noise of baseline (b1/b2/b3:
  21.7/20.6/20.9 vs b0 21.3) — at this scale their real effect is negligible, not negative.

## 15. THE SIMPLE CONFIG — production-validated (job 3443611)

```sh
export CHUNK_EXTRA_STOP=']]'              # contradiction / cot_mode=none only (§11b caveat)
export CHUNK_MAX_BATCHED_TOKENS=<longest prompt, rounded up>   # 4096 @2560, 9216 @8192
export CHUNK_MAX_NUM_SEQS=16
export CHUNK_SEQ_HEADROOM=18
# NO fp, NO cache — not needed.
```

4B production checkpoints, eval_size=500, gate = f1 within 0.018 of known-good AND parse_rate 1.0:

| rung | prod chunked | simple config | speedup | f1 (known-good) | verdict |
|---|---|---|---|---|---|
| 2560 | 1162.7s | **95.8s** | **12.1x** | 0.8620 (0.8613, +0.0007) | REPRODUCES |
| 8192 | 2066.1s | **223.5s** | **9.3x** | 0.8012 (0.8039, −0.0027) | REPRODUCES |

223.5s also beats round 3's best-ever 311.9s (fp+cache+seqs64+stop) by 1.4x with fewer knobs.
Chunked now stands ~2.9x slower than dense+stop (77.8s at 8k), down from 17.8x.

## 16. The varlen rewrite (`CHUNK_VARLEN_PREFILL=1`) — implemented, exact, and modestly faster

§10 declared the varlen decomposition "written from scratch" work. Two of its blockers turned out
to already exist inside vLLM 0.25.1: `vllm/v1/attention/ops/merge_attn_states.py` (the LSE merge,
used by cascade attention) and the bundled `vllm_flash_attn.flash_attn_varlen_func` (which also
does paged KV via `block_table`). What remained was the decomposition itself:

    A. causal attention within each maximal constant-chunk-id run  -> flash varlen, cu_seqlens
    B. FREE q -> everything strictly before its own run  (thin fp32 strip)
    C. doc  q -> every FREE token strictly before it     (thin fp32 strip)
    decode: generated tokens are FREE -> plain causal    -> paged flash, block_table

Every allowed (q,kv) pair is covered exactly once, so LSE-merging A with (B|C) reproduces the
full rule. Proof + float64 elementwise test over 9 layouts (adjacent docs, unmatched markers,
multi-request packing...): `test_varlen_decomposition.py`, max|err| ~1e-16.

On an eligible step (every request either a one-shot full prefill or an all-FREE-query
continuation) NO chunked BlockMask is built, the flex kernel never runs, and the whole-KV-cache
`transpose().contiguous()` re-layout in the impl patch is skipped. Ineligible (mixed/partial)
steps fall back to the historical flex path, bit-identical. `enable_prefix_caching` is turned off
by the driver under this flag so prefix-cache hits don't demote prefills to the fallback.

Small-scale, min over 3 back-to-back reps (job 3443713, co-tenant idle):

| arm | min gen_s |
|---|---|
| simple config | 15.4 |
| + varlen | 14.3 |
| + varlen, seqs64/headroom66 | **13.0** |

Stage profile (job 3443695): `varlen_fwd_pf` 0.66ms/call, `varlen_fwd_dec` 0.29ms/call,
plan build 0.6ms/step — the varlen machinery is ~free; the gain is capped at this rung because
2.5k-token mask builds were already cheap and 38-62 mixed steps still take the flex fallback.
The mask-build term varlen deletes grows with rung length x cache size, so the expected payoff
is at 8k/32k — and it is (job 3443769, 4B, eval_size=500, same gates):

| arm | gen_s | vs prod chunked | vs simple config | f1 (known-good) | verdict |
|---|---|---|---|---|---|
| varlen, seqs16 @2560 | **59.8s** | **19.5x** | 1.60x | 0.8627 (0.8613, +0.0014) | REPRODUCES |
| varlen, seqs16 @8192 | **133.7s** | **15.5x** | 1.67x | 0.8019 (0.8039, −0.0020) | REPRODUCES |
| varlen, seqs64 @8192 | 209.5s | 9.9x | 1.07x | 0.8019 | REPRODUCES, but see below |

The seqs64 arm is the coverage lesson in one row: at high concurrency nearly every step mixes
decodes with a PARTIAL prefill, so only 31/486 steps took the varlen path (vs 719/903 at seqs16)
and the flex fallback ate the gain. **Production setting: seqs16.**

Correctness at small scale: parse_rate 1.0, f1 identical, gen_match 95/100 vs the flex path
(the 5 diverge at flash-vs-flex numerics, the same class as round 2's batch-composition effects).

## 17. CONFOUND: cubbins was oversubscribed the whole time

`sacct` shows a co-tenant job holding `gres/gpu=8` on cubbins from 01:50 through every round-5
bench. Slurm scheduled our 1-GPU jobs onto the same node anyway, i.e. GPU sharing, and the noise
it injects is not symmetric jitter — it produced two complete phantoms:

* **a4/a5's "fp+cache interaction" (87.0s / 41.4s) is NOT real.** Each knob alone measured clean
  (b1/b2/b3), and the combination was never re-measured before the story "fp x cache interact
  pathologically" was written. Retract it.
* **c1_varlen's first measurement (134.9s) was 9.4x contention noise.** The identical config
  measured 15.4s under CHUNK_PROFILE=1 minutes later and 14.3/14.4/14.3 in three clean reps.

Protocol change for everything after 5d: repeat arms back-to-back, judge by MIN, and log
`nvidia-smi --query-compute-apps` at arm start. A single-shot small-scale timing on a shared
node is not evidence.

(The production validations are unaffected where it matters: f1 gates are correctness, and
contention can only make the measured SPEEDUPS conservative.)

## 18. Where this leaves the ledger

| | 8k contradiction, 500 ex | vs dense+stop 77.8s |
|---|---|---|
| round-0 production chunked | 2066.1s | 26.6x slower |
| round-3 best (fp+cache+seqs64+stop) | 311.9s | 4.0x |
| **round-5 simple config** | **223.5s** | **2.9x** |
| **round-5 + varlen (seqs16)** | **133.7s** | **1.7x** |

Open items, in order of leverage:

1. ~~Varlen production numbers~~ — DONE: 133.7s at 8k (1.67x over the simple config), gates
   pass. Next: re-gate the 16k/32k rungs (RUNG_GATE.tsv) with varlen+seqs16; the mask-cost
   argument says the margin should widen further there.
2. **Mixed-step coverage** — 38-62 steps per run still take the flex fallback because a step
   mixes decodes with a prompt's one-shot prefill. Handling partial prefills in the varlen path
   (per-request split is already implemented; what's missing is doc-boundary state for a prompt
   split across steps) would retire the flex path entirely.
3. **Decode fixed cost** — unresolved by construction (§11c still applies): eager mode + per-step
   python. Matters only for long-generation (CoT) evals; the `]]`-class early stop is what keeps
   it small for the CTC suite.

Artifacts: `smallscale_bench.sbatch` (5a), `smallscale_isolate.sbatch` (5b),
`validate_simple.sbatch` (5c, job 3443611), `smallscale_varlen.sbatch` (5c', job 3443669),
`smallscale_varlen_diag.sbatch` (5d, job 3443695), `smallscale_varlen_rerun.sbatch` (5e, job
3443713), `validate_varlen.sbatch` (5f, job 3443769), `test_varlen_decomposition.py`.
New env knobs (all default OFF/historical): `CHUNK_VARLEN_PREFILL`, `CHUNK_VARLEN_DECODE`.
