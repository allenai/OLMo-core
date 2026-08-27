# Making sparse-landmark native evals fast

Follow-on to `README.md` in this directory. That one measured a 3.2x-faster decode for
**`FastLandmarkAttention`** and left it as a standalone monkeypatch nobody called. This one wires a
sparse decode into the actual generation path, extends it to the *other* landmark family
(**`SparseLandmarkAttention`**, `AttentionType.sparse_landmark` — the thing the CTC sparse-landmark
checkpoints actually use), and unlocks batch size > 1.

⚠ **Everything here is validated on CPU only. Nothing in this note has been run on a GPU.** The
numerics are pinned by exact-parity tests against the shipped decode (details below); the *speedup*
is an argument from the op count and memory traffic, not a measurement. See
[What still needs GPU validation](#what-still-needs-gpu-validation).

---

## The confusion this note has to clear up first

Two different things are called "sparse landmark", and they are not the same code path:

| name | file | what it is |
| --- | --- | --- |
| `FastLandmarkAttention` | `landmark_fast.py` | landmark attention with a grouped softmax: a block gate over its landmark, then a within-block softmax over the block's content |
| `SparseLandmarkAttention` | `landmark_sparse.py` | a *different architecture*: full causal attention inside a chunk, past chunks visible **only** through their last `num_landmarks` tokens. One flat softmax, no block content across chunks |

`landmark_sparse_decode.py` (from the previous session) implemented a fast decode for the **first**
one. `enable_sparse_decode(model)` on a sparse-landmark model raised
`RuntimeError: no landmark attention layers found`. So the "3.2x already exists" starting premise
was true for `fast_landmark` and did not apply to `sparse_landmark` at all.

## Why the sparse-landmark decode was slow

`SparseLandmarkAttention._forward_generate` at `T == 1`:

```python
kh = repeat_kv(kvm.k_cache[:, :total].transpose(1, 2), n_rep)   # materializes a 4x copy of the cache
vh = repeat_kv(kvm.v_cache[:, :total].transpose(1, 2), n_rep)
att = self._decode_one(qh, kh, vh, start_pos)                   # scores all `total` keys, masks ~99% to -inf
```

Both pathologies from the original README, and the ratio is *worse* here than for fast-landmark. At
32k with the shipped `mem_freq=63` (`block_size = 64`, `num_landmarks = 1`), a decode query is
allowed to see:

* its local section — 64 tokens of its own chunk, growing by one per generated token, and
* one landmark key per past chunk — 512 keys at 32k, and only `top_k` of those (51 at the default
  `landmark_top_k_fraction=0.1`) survive retrieval.

That is **~600 keys out of 32768**, or ~115 with top-k. The shipped decode reads all 32768 of them,
at 4x width because of `repeat_kv`, and throws >98% away. Key/value bytes touched per layer per
token drop by roughly `n_rep * total / (n_lm + local)` ≈ **200x** in the sparse form. (Wall clock
will not drop 200x — the original README's finding stands: at these sizes the decode stops being
memory-bound and becomes launch-bound at ~20 torch ops per layer. Expect the sparse-landmark decode
to land in the same "flat across context length, ~10 ms/token" regime the fast-landmark one did.)

---

## What changed

### 1. A genuinely sparse decode for `SparseLandmarkAttention`

`src/olmo_core/nn/attention/landmark_sparse_decode.py` (+~380 lines, second half of the module):

* `landmark_positions(section_start, block_size, num_landmarks, device)` — the landmark key set
  built in `O(n_landmarks)` instead of materializing an `O(section_start)` boolean mask.
* `landmark_chunk_count(...)` — the exact `n_chunks` the shipped top-k uses. Note this is **not**
  `ceil(section_start / L)`: a trailing partial chunk counts only if the section boundary actually
  reaches its landmarks. Getting this wrong silently changes the `n_chunks <= top_k` early-out.
* `sparse_chunk_decode(...)` — the decode: gather the past chunks' landmark rows once, score them,
  apply top-k chunk retrieval, score the contiguous local section, one flat softmax over the union,
  two GQA-folded `A @ V` matmuls. No `repeat_kv` anywhere (the query is folded into the KV-head
  grouping by the existing `_gqa_scores` / `_gqa_av` helpers).
* `sparse_chunk_decode_ragged(...)` — the same thing for a right-padded cross-length batch, with
  per-row section start / query position / top-k.
* `_forward_generate_sparse_chunk` — the patched `_forward_generate`. Prefill is byte-for-byte the
  shipped path (it still delegates to `_prefill`, so this composes with the prefill-topk module).

`enable_sparse_decode(model)` now patches **both** families and takes `strict=False` for callers
that just want "make it fast if it applies". `disable_sparse_decode` / `reset_sparse_decode_cache`
round it out.

**Selection is shared, not reimplemented.** `landmark_sparse.py` gained
`landmark_chunk_topk_keep(lm_scores, chunk_ids, n_chunks, top_k)`, factored *out of*
`SparseLandmarkAttention._apply_topk_landmark_retrieval` — the shipped path and the sparse path now
call the same function, so which chunks get retrieved is identical by construction rather than by
inspection. Top-k is a hard discrete decision; a tolerance test would not have caught a drift there.

### 2. Wired into the generation path, on by default

* `GenerationConfig.landmark_sparse_decode: bool = True` (new field, documented).
* Environment escape hatch `OLMO_LANDMARK_SPARSE_DECODE=0`, which wins over the config field — so a
  suspicious eval number can be A/B'd against the shipped decode without touching any driver.
* `TransformerGenerationModule._maybe_enable_sparse_decode()` installs it lazily on the first
  landmark generate (idempotent, `strict=False`, wrapped in a try/except so an optimization can
  never break generation), from both `generate_batch` and `generate_landmark_batch`.

Nothing in the eval drivers needs to change. `eval_lc_native_landmark.py` picks this up as-is.

### 3. batch_size > 1 for sparse-landmark generation

**Why it was bs=1.** Not a KV-cache-manager limitation and not exact-length *bucketing* per se — the
root cause is that chunk boundaries are tied to **absolute position**. A left-padded row would have
its content start at position `pad_len`, putting every chunk boundary and every landmark at the
wrong offset relative to training. So `generate_batch` rejects any non-trivial `attention_mask` and
`_forward_generate` rejects non-zero `cache_leftpad`; only prompts of *exactly* equal length can
share a batch, which on a variable-length eval means an effective batch size of ~1.

**The fix already existed for the other family.** `FastLandmarkAttention` solved this with a
right-padded ragged decode (`_supports_ragged_decode`, `_decode_ragged`,
`TransformerGenerationModule.generate_landmark_batch`): right-padding is legal because each row's
content still starts at position 0 and only the pad *tail* differs, and the tail is causally future
during prefill. `SparseLandmarkAttention` simply never implemented the hooks, so
`supports_landmark_ragged_batch()` returned False and `eval_lc_native_landmark.py` silently fell
into its slow `else` branch.

Implemented on `SparseLandmarkAttention` (`landmark_sparse.py`, +~150 lines), mirroring the
fast-landmark structure:

* `_supports_ragged_decode = True`, `set_landmark_ragged_decode` / `set_ragged_qpos` /
  `clear_ragged_decode`, `_ragged_section_start`, `_forward_generate_ragged`, `_decode_ragged`,
  `_decode_topk_ragged` (per-row chunk budget via the argsort-rank trick, since `topk` takes a
  scalar `k`).
* The generation module's `generate_landmark_batch` no longer hardcodes `block_size = mem_freq + 1`;
  it reads the layer's own `block_size`.

Plus the two decodes compose: with the patch installed, ragged steps go through
`sparse_chunk_decode_ragged`, so you get batching **and** sparsity rather than choosing.

Consequence: `eval_lc_native_landmark.py --no-fast-batch` is now the only way to get the old
bs≈1 path for a sparse-landmark model; by default it will batch.

### 4. A silent-wrongness guard (`num_landmarks > 1`)

`_insert_landmark_tokens` puts exactly **one** landmark at the end of each block. A
`SparseLandmarkAttention` configured with `num_landmarks > 1` has `block_size = mem_freq +
num_landmarks`, so that prompt's landmarks land at the wrong positions — and it would have read as
a modeling result, not a harness bug. `TransformerGenerationModule._landmark_block_size` now
rejects it with a clear error. All shipped sparse-landmark runs use `num_landmarks=1`
(`MEM_FREQ_SPARSE = 63`), so nothing real is blocked.

---

## Files changed

| file | change |
| --- | --- |
| `src/olmo_core/nn/attention/landmark_sparse_decode.py` | sparse-landmark decode (scalar + ragged), both-family `enable_sparse_decode`, index/count helpers |
| `src/olmo_core/nn/attention/landmark_sparse.py` | `landmark_chunk_topk_keep` factored out; ragged decode support |
| `src/olmo_core/generate/generation_module/config.py` | `GenerationConfig.landmark_sparse_decode` |
| `src/olmo_core/generate/generation_module/transformer/generation_module.py` | install hook + env override, generic block size, `num_landmarks` guard |
| `src/test/nn/attention/landmark_sparse_decode_test.py` | **new** — sparse decode parity (55 cases) |
| `src/test/nn/attention/landmark_sparse_ragged_decode_test.py` | **new** — ragged parity, dense and sparse (21 cases) |
| `src/test/generate/generation_module/transformer/landmark_generation_test.py` | end-to-end generation parity, env escape hatch, ragged batch parity, `num_landmarks` guard |

---

## What is actually validated (CPU, no GPU)

The sparse mixer's decode math is pure torch, so the parity gates run for real on CPU:

* **`sparse_chunk_decode` vs the shipped `_decode_one`**, over the cross product of decode mode
  (`extend_last_block` / `generation_only`) x `top_k` (`None`/1/2/100) x GQA ratio (1x/2x/4x) x
  `num_landmarks` (1/2), at 7 query positions each covering both query regimes (prompt-position
  per-chunk rule, generated one-long-local-block rule) and a landmark-position query.
* **Selection equality** under top-k, read off with one-hot values: the *support* of the output
  weights matches exactly, and exactly `top_k` past chunks are retrieved.
* **`landmark_positions`** against the brute-force boolean mask for every section boundary in
  `[0, 3L]` at three `(L, G)` settings — including boundaries inside a chunk.
* **End-to-end through `_forward_generate`**: prefill + 5 decode steps, patched vs unpatched, and
  `disable_sparse_decode` restores the original bitwise.
* **Full-model generation**: `generate_batch` with the sparse decode on produces **token-identical**
  output to the shipped decode, on a 2-layer sparse-landmark transformer, in both decode modes, over
  two successive calls (which is what catches a stale hoisted landmark cache).
* **Ragged decode**: every row of a batched decode equals the scalar bs=1 `_decode_one` on that row
  alone — for the dense ragged decode and the sparse ragged decode, with per-row prompt lengths,
  per-row top-k (including a row whose budget exceeds its chunk count), GQA, and `num_landmarks=3`.
* **Ragged generation**: `generate_landmark_batch` on 4 prompts of lengths 11/26/19/40 matches each
  row's own bs=1 `generate_batch` output.

Run them with:

```bash
/scratch/users/prasann/conda/envs/corpus-reasoning-olmo/bin/python -m pytest -q \
  src/test/nn/attention/landmark_sparse_decode_test.py \
  src/test/nn/attention/landmark_sparse_ragged_decode_test.py \
  src/test/generate/generation_module/transformer/landmark_generation_test.py
```

Agreement is to floating-point reassociation (`rtol=atol=1e-5` in float32), not bitwise: the sparse
path splits the softmax denominator and the `A @ V` reduction into a landmark term and a local term.
Selection tests are exact.

## What still needs GPU validation

Nothing below is a known problem — it is the list of things CPU tests structurally cannot cover.

1. **The actual speedup.** Not measured anywhere. `bench_decode.py` in this directory benchmarks the
   *fast-landmark* decode; it has no sparse-landmark arm. First GPU task: add one and get the
   ms/token and GB/token table at 8k/16k/32k, bs=1 and bs=8, against the shipped decode. Until that
   exists, "substantially faster" is a prediction.
2. **bf16 agreement.** All parity is float32-on-CPU. In bf16 the two-term softmax split and the
   GQA-folded matmul can differ from the dense form by more than fp32 reassociation. The
   fast-landmark equivalent was validated "to bf16 noise" on GPU across 21 cases; the sparse-landmark
   one has not been. **Do this before trusting any eval number**: run one real checkpoint, one task,
   ~50 examples, with `OLMO_LANDMARK_SPARSE_DECODE=0` and `=1`, and compare generated text.
3. **cuBLAS kernel selection.** `_gqa_scores` reshapes the query to `(B, H_kv, n_rep, D)` and matmuls
   against the unexpanded cache. Same dot products, different shapes — so cuBLAS may pick a different
   kernel with a different reduction order than the dense `(B, H, 1, D) x (B, H, total, D)` form.
   Expected to be within bf16 noise; unverified.
4. **The ragged batch at scale.** `generate_landmark_batch` was only exercised at B=4 with ≤40-token
   prompts. Memory behaviour at eval-realistic shapes (B=8-16, 32k prompts) is untested — in
   particular `sparse_chunk_decode_ragged`'s local-window gather allocates `(B, H_kv, W, D)` where
   `W = max(qpos - section_start) + 1` grows with generated length.
5. **Interaction with the Triton training kernel.** Prefill still runs `_attn_core`, which uses the
   fused Triton kernel on CUDA. Unchanged by this work, but the patched `_forward_generate` is the
   caller now, so the prefill path should be smoke-tested on GPU once.
6. **`mypy`** is not installed in `corpus-reasoning-olmo`, so `make type-check` was not run. `ruff`
   and `black --check` pass on every changed file.

---

## Scoping: serving sparse-landmark Qwen3.5 in vLLM 0.25.1

**Not implemented. This is an estimate.**

### The good news: it is a pure positional mask

The sparse-landmark rule is
`attend(q, kv) = (same_chunk(q, kv) & causal) | (kv_is_landmark & kv_chunk < q_chunk)`, and
landmarks sit at fixed periodic positions (`pos % 64 == 63`). So unlike the document-chunked mask —
which needs per-request `chunk_ids` derived by scanning token ids for `<|doc_start|>`/`<|doc_end|>`
every step — the landmark mask needs **no token inspection at all**, only `block_size` and a
per-request prompt length. It is strictly *simpler* than a mask we have already shipped.

That matters because the integration point already exists and is battle-tested:
`src/scripts/ctc_eval/lib/vllm_chunked_patch.py` injects a custom rule into vLLM's **FlexAttention**
backend via its `logical_mask_mod` hook, and `run_vllm_eval_generic.py` selects it with
`attention_config={"backend": "FLEX_ATTENTION", "flex_attn_kv_block_size": 32}`.

**No custom model class and no new attention backend are required.** The weights are stock
Qwen3.5; only the mask differs.

### Concrete integration points

1. **The mask_mod** — replace `_build_chunked_final_mask_mod` with a landmark version. In vLLM's
   logical coordinates this is ~10 lines:

   ```python
   sec = torch.where(lq >= P[req], (P[req] // L) * L, (lq // L) * L)
   is_lm = (lkv % L) >= (L - G)
   ok = ((lkv >= sec) & (lkv <= lq)) | (is_lm & (lkv < sec))
   ```
   `P` is the per-request landmark-prompt length, available from `input_batch.num_prompt_tokens` via
   the thread-local hook the chunked patch already installs.

2. **Reuse, unchanged, four of the chunked patch's five monkeypatches.** These were hard-won for the
   GDN hybrid and are not landmark-specific: the runner hook that parks `input_batch` in a
   thread-local, the `.view` → `.reshape` fix (vLLM pads the attention page to align with the mamba
   page, breaking the zero-copy view), the BLOCK_N power-of-2 clamp (the hybrid's 288-token page trips
   `tl.arange`'s power-of-2 requirement), and the `load_model` extras hook. Only the mask rule is new.

3. **Prompt-side landmark insertion.** The harness must insert token id `248200` (Qwen3.5;
   `document_chunk_landmark.py` is canonical) every 63 content tokens before the prompt goes in, and
   strip landmarks from the output. The id sits in the *untrained padded* region below the embedding
   matrix's row count, so **no vocab resize and no embedding surgery** — but the HF tokenizer will
   not produce it, so prompts must be fed as `TokensPrompt` (which the driver already does). The
   checkpoint must be one whose landmark row was repaired.

4. **The full seven-piece Qwen3.5 serving-copy recipe still applies** unchanged (VL wrapper config,
   `model.*` → `model.language_model.*` rename, dummy `visual.*` params, arch override,
   `limit_mm_per_prompt`). Landmark support does not interact with it. Look for an existing
   `/data/prasann/ctc_suite/vllm_serving_4b_v3/<ckpt>/` before rebuilding.

### Chunked prefill

**Compatible in principle, and better than our native path.** The note in the task premise —
"`landmark_sparse.py`'s generation path only supports single-shot prefill from position 0" — is a
limitation of *our* KV-cache code (it raises `NotImplementedError` for `T > 1` with a non-empty
cache), not of the attention pattern. The mask is a pure function of absolute positions, so a
prefill split across chunks, prefix caching, and paged KV all work as long as the mask_mod is
evaluated in logical coordinates — which is exactly what the FlexAttention hook provides. Worth
verifying rather than assuming: the chunked patch runs today with
`CHUNK_VARLEN_PREFILL=1`, so there is a working reference for prefill splitting under a custom mask.

### The two honest blockers

1. **Top-k retrieval cannot be expressed as a mask.** `mask_mod` is a function of indices only;
   top-k chunk retrieval is a function of *scores*. Our sparse-landmark evals default to
   `landmark_top_k_fraction=0.1`, so a FlexAttention mask_mod would serve the **dense-gating**
   variant (`top_k=None`) and its numbers would not be comparable to the native ones. Options:
   (a) re-run both sides at `top_k=None` and state it; (b) accept a mask-only vLLM path for
   throughput-oriented work and keep the native path as the scoring path; (c) write a real custom
   backend that does two-stage sparse decode against the paged KV cache — a much larger job (see
   below). Decide this *before* starting, because it determines whether the cheap path is even
   useful.
2. **FlexAttention gives correctness, not much speed.** BlockMask sparsity is exploited at
   `flex_attn_kv_block_size` granularity (32 in our config). With `L=64, G=1`, a landmark lands in
   every *second* 32-wide KV block, so only about half the cross-chunk key blocks can be skipped —
   ~2x, against a mask whose true sparsity is ~30-60x. And FlexAttention is itself slower than
   FlashAttention per unattended-block. Precedent: the chunked eval was 17.8x slower than dense
   before tuning and 1.7x after. Expect a similar ballpark, i.e. **vLLM would buy throughput from
   continuous batching and paged memory, not from the sparsity**.

### Effort estimate

| scope | what you get | estimate |
| --- | --- | --- |
| **Mask-only FlexAttention path** (mask_mod + landmark insertion in `build_prefills.py` + a serving-copy sanity pass), `top_k=None` semantics | correct sparse-landmark serving, vLLM throughput, numbers **not** comparable to top-k native evals | **1-2 days**, most of it validating against a native run rather than writing code |
| **Above + parity harness** (native `top_k=None` reference run, ~50 examples, generated-text diff) | a number you can defend | **+0.5-1 day** |
| **Custom paged sparse-decode backend** (landmark scoring + top-k + gather against vLLM's paged KV cache, per layer, plus a metadata builder) | true sparsity and top-k parity in vLLM | **1-2 weeks**, and it duplicates the Triton fusion work the original README already scoped as "not yet written" |

**Recommendation.** Do the mask-only path only if the throughput of continuous batching is the goal
and `top_k=None` semantics are acceptable. Otherwise the higher-value next step is the one the
original README named: fuse stage B of the native decode into a single Triton kernel. That is the
same work the custom vLLM backend would need anyway, it lands in the path our evals already use, and
with #3 above (ragged batching) the native path now has continuous-batching-like utilization too.
