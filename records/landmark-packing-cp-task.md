# Task: implement Context Parallelism for landmark attention with sequence packing

## Summary
`FastLandmarkAttention` and `SparseLandmarkAttention` (OLMo-core, branch `prasann/landmark`)
support **Ulysses context parallelism (CP)** *and* **sequence packing with intra-document masking
(`cu_doc_lens`)** — but **not the two together**. The combination is currently blocked by an
explicit `NotImplementedError`. Implement and validate the combination so packed landmark SFT can
use CP to shard activations (needed to fit 32k–64k sequence lengths in GPU memory; landmark
packing currently must run FSDP-only without CP, which OOMs at long context because activations
aren't sequence-sharded).

## Current state / where it's blocked
- `src/olmo_core/nn/attention/landmark_fast.py`, `FastLandmarkAttention.forward` (~line 701):
  ```python
  if cu_doc_lens is not None and self.cp_enabled:
      raise NotImplementedError(
          "Intra-document packing (cu_doc_lens) is not supported together with context parallelism."
      )
  ```
- The same guard exists in `src/olmo_core/nn/attention/landmark_sparse.py`.
- Packing was added in commit `cf4fa5b9` ("Add sequence-packing (intra-document masking) to
  landmark attention"). CP support predates it. Only the *intersection* was left unimplemented.

## How the forward works today (single-doc + CP, OR packing without CP)
In `FastLandmarkAttention.forward` (landmark_fast.py ~672–754):
1. `q,k,v = self._prepare_qkv(x, pos_*, cu_doc_lens=cu_doc_lens)` — applies RoPE on the **local**
   sequence shard `x` of shape `(B, T_local, …)`. Per-document RoPE position reset is done here via
   `cu_doc_lens` (RotaryEmbedding intra-doc RoPE).
2. If `cp_enabled`: `all_to_all_single_cp2hp(q)` / `all_to_all_cp2hp([k,v])` — Ulysses all-to-all
   that **gathers the full sequence `T`** per rank and shards heads (`n_heads/cp`).
3. `doc_id = build_block_doc_id(cu_doc_lens, B, T, block_size)` — per-landmark-block document id over
   the **full** gathered sequence `T` (see `src/olmo_core/nn/attention/landmark.py:26`).
4. `att = self._attn_core(q, k, v, doc_id=doc_id)` → `fused_landmark_attention_fast(..., doc_id=doc_id)`
   (the Triton kernel masks cross-document attention using `doc_id`).
5. If `cp_enabled`: `all_to_all_single_hp2cp(att)` — scatter sequence back, gather heads.

## Why it's non-trivial (the core problem)
The packing needs document boundaries in **two frames**:
- **RoPE per-doc reset** (step 1) runs on the **local** `T_local` shard → needs *local* `cu_doc_lens`.
- **Kernel `doc_id`** (step 3) runs on the **full** gathered `T` → needs *full* `cu_doc_lens`.

And because Ulysses shards the sequence **uniformly** (`T_local = T/cp`, contiguous per rank), a
packed document can **straddle a CP-rank boundary**, so per-doc RoPE reset at a shard edge may need a
document's start position that lives on another rank.

## Implementation (Approach A — general; handle docs spanning rank boundaries)
1. **Remove the guard** in both `landmark_fast.py` and `landmark_sparse.py`.
2. **`doc_id` (post-gather):** `build_block_doc_id` already takes full-frame `cu_doc_lens` and the
   gathered `T`; confirm the `cu_doc_lens` reaching `forward` is the **full-sequence** cumulative doc
   lengths (flattened over batch; `[0, …, B*T]`). If the train module shards/relabels `cu_doc_lens`
   per CP rank, reconstruct/pass the full version for `doc_id`. (Step 3 then works unchanged.)
3. **RoPE per-doc reset (pre-gather, local shard):** make the per-document position reset correct for
   the rank's local slice, including documents that start on a previous rank — i.e. the RoPE position
   within a doc must be the token's offset from the doc's **global** start, even when that start is on
   another rank. Options: pass a precomputed per-token position tensor (already doc-reset) sliced per
   CP rank instead of deriving the reset from local `cu_doc_lens`; or compute the local doc offsets
   from the full `cu_doc_lens` + the rank's global sequence offset. Verify against how the non-packed
   landmark CP path and the flash/ring CP path obtain positions.
4. Mirror all of the above in `SparseLandmarkAttention`.
5. Keep block-alignment invariants: `T` (and `T_local`) multiples of `block_size = mem_freq + 1`;
   documents are block-aligned by `LandmarkPackingInstanceSource`
   (`src/olmo_core/data/composable/landmark_packing_instance_source.py`).

Relevant helpers: `build_block_doc_id` (`landmark.py:26`), the CP all-to-all functions
(`olmo_core.distributed.parallel.context_parallel`: `all_to_all_cp2hp`,
`all_to_all_single_cp2hp`, `all_to_all_single_hp2cp`), `apply_cp` (Ulysses-only) in each landmark
attention class, and the Triton kernel `doc_id` path (`_FusedLandmarkAttentionFast` forward,
`landmark_fast.py:370`).

## Mandatory correctness validation (this is the whole point of using a GPU box)
A wrong mask here fails **silently** (corrupted training, not a crash), so prove correctness:
1. **Unit/numeric:** on a single GPU vs multi-GPU CP group, run the packed landmark attention on the
   same `(q,k,v, cu_doc_lens)` and assert the CP output matches the non-CP output (allclose, bf16
   tolerance). Cover: docs that align to rank boundaries AND docs that straddle them; ≥2 and ≥4 CP
   degrees; both fast and sparse variants.
2. **End-to-end loss match:** train a few steps of the unified fast-landmark SFT on the 1k debug
   shards with CP **off** vs CP **on** at the same seed/data and assert the per-step loss matches
   (within bf16 noise). Script: `src/scripts/train/sft/Qwen3-4B-fast-landmark-unified-SFT.py`
   (it has `cp_config` omitted today; add a CP path / override). Debug data already exists at
   `/weka/oe-training-default/ai2-llm/checkpoints/prasanns/suite_it_sft_qwen/combined_debug`
   (token_ids_part_*.npy + labels_mask_*.npy, Qwen3-0.6B-tokenized, 967 instances). Use
   `UNIFIED_SEQ_LEN` to pick a CP-divisible seq length, e.g. 16384 with cp degree 8 (T_local=2048,
   a multiple of block_size 64).
3. Add a regression test under `src/test/nn/attention/` alongside the existing
   `landmark_fast_kernel_test.py` / `landmark_sparse_kernel_test.py`.

## Acceptance criteria
- Both landmark variants run with `cu_doc_lens` + Ulysses CP (no `NotImplementedError`).
- CP output is numerically equal to non-CP for packed inputs, including boundary-straddling docs.
- End-to-end CP-vs-no-CP loss matches on the debug set.
- 64k packed fast-landmark SFT fits and trains with CP (e.g. cp degree 8) where FSDP-only OOMs.
- New GPU tests pass; `make checks` clean.

## Context / starting point
- Branch `prasann/landmark` (current HEAD). The FSDP-only packed path already works end-to-end
  (data → `LandmarkPackingInstanceSource` → train → checkpoint) — validated on the 1k debug set at
  8k seq len, 1 node, no CP. This task adds the CP path on top.
- Reference packed (no-CP) scripts: `src/scripts/train/sft/Qwen3-4B-{fast,sparse}-landmark-packed-SFT.py`
  and the unified variants `Qwen3-4B-{fast,sparse}-landmark-unified-SFT.py`.
- The dense variant uses flash + Ulysses/ring CP already (for reference on how CP + intra-doc
  masking is wired): `Qwen3-4B-dense-unified-SFT.py`, and `Attention.forward` /
  `RingContextParallelStyle` handling of `cu_doc_lens_q`/`cu_doc_lens_k`/`local_k_slice`.
