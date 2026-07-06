# Task: make landmark top-k decode *sparse* (O(k·block) instead of O(context) per token)

## Summary
Landmark-attention models (OLMo-core, branch `prasann/landmark`) now do **hard top-k landmark block
retrieval at decode by default** (top ~10% of blocks; `GenerationConfig.landmark_top_k_fraction`,
default 0.1). But the decode is only top-k *numerically*, not *computationally*: at each generated
token it still computes attention over the **entire** KV cache and then **masks** the non-selected
blocks to zero. So long-context eval is O(context) per token and very slow (e.g. ~22 s/example at
32k, `bs=1`). Make the decode **actually sparse**: score the cached landmark keys, pick the top-k
blocks, then **gather and attend over only those blocks' content KV + the local block** — O(num_blocks
+ k·block_size) per token. This is the landmark paper's inference-efficiency claim (Mohtashami &
Jaggi 2023, §3.2) and should give a large speedup at long context with **identical outputs**.

## Current (dense-mask) decode — where it is
The per-token decode for each variant computes `scores = q @ K_all^T` over the full cache, calls
`_apply_topk_landmark_retrieval` (which sets non-top-k *landmark* scores to `-inf`), then
`landmark_grouped_softmax`, then `probs @ V_all`:

- `src/olmo_core/nn/attention/landmark_fast.py`: `_decode_one` (~885), `_decode_one_eval` (~925),
  `_apply_topk_landmark_retrieval` (~857), `_forward_generate` (~791).
- `src/olmo_core/nn/attention/landmark_sparse.py`: `_decode_one` (~488), `_apply_topk_landmark_retrieval`
  (~528), `_forward_generate` (~419).
- `src/olmo_core/nn/attention/landmark_compressive.py`: `_decode_one` (~757), `_decode_one_eval` (~776).
- Shared: `landmark_grouped_softmax` (`landmark.py:197`). Top-k count comes from
  `set_landmark_eval_decode(prompt_len, mode, top_k=...)`; the generation module already passes the
  computed `k` (`generation_module/transformer/generation_module.py`, the `_set_landmark_eval_decode`
  call site).

The KV cache holds all past keys/values (`KVCacheManager`); landmark (memory) tokens sit at periodic
positions `pos % block_size == block_size-1` (`block_size = mem_freq + 1`).

## What to implement
At each decode step (query at absolute `qpos`):
1. **Score landmarks only:** `q · K_landmark` over the cached landmark keys → one score per past block.
   O(num_blocks).
2. **Select top-k blocks** per (batch, head) — the same selection `_apply_topk_landmark_retrieval`
   makes today (keep the existing tie-break / `<= top_k` short-circuit semantics so results match).
3. **Gather** only the selected blocks' **content** K/V (each block is `mem_freq` content tokens +
   1 landmark) plus the **local** block (the growing "one long local block" for generated tokens; see
   `_decode_one_eval` / `landmark_decode_mode`). Index the cache to build a compact
   `(B, H, k*block + local, D)` K/V.
4. **Attend** over only those keys via the same grouped-softmax math (landmark gates each selected
   block, content renormalizes within block, local block is dense) so the output equals today's
   masked result. Avoid materializing the full `(B,H,1,context)` score tensor.

Do this for **all three variants** (fast, sparse, compressive), preserving each one's grouped-softmax
/ landmark-gating semantics (sparse uses chunked masking; compressive has the
`landmark_nonselected_mass` reserve — keep it correct). Prefill is unchanged (single-shot dense).

## Mandatory validation (correctness is silent if wrong)
1. **Numeric equivalence:** for each variant, assert the new sparse `_decode_one` output is
   `allclose` (bf16 tol) to the **current dense-mask top-k** `_decode_one` output, across: several
   `top_k` values, landmark vs non-landmark query positions, the `extend_last_block` and
   `generation_only` decode modes, and a query whose true needle is in a non-top-1 block. Add tests
   beside `src/test/nn/attention/landmark_*_kernel_test.py`.
2. **End-to-end score match:** run a few `cr_ruler_*` / `longctx_*` evals on a real landmark
   checkpoint (e.g. `q4b-fast-landmark-dolma3longmino/step2385`) before vs after this change and
   confirm the **scores are identical** (same generations) — only faster.
3. **Speed:** report tokens/s (or s/example) at 8k/32k/64k before vs after; expect a large drop at
   long context (the whole point).

## Acceptance criteria
- Sparse decode for fast/sparse/compressive landmark; outputs bit/`allclose`-equal to the current
  dense-mask top-k decode.
- Eval scores unchanged; measured large speedup at 32k/64k.
- New tests pass; `make checks` clean.

## Context
- Branch `prasann/landmark` (HEAD). Top-k retrieval is now on by default (commit added
  `landmark_top_k_fraction=0.1`); this task makes that retrieval *cheap*. It does NOT change which
  blocks are selected or the outputs — purely a compute/memory optimization of the decode.
- Eval path: oe-eval `prasann/longctx-eval`, olmo_core backend (`eleuther_olmo_core.py`),
  `bs=1` landmark decode. Launchers: `launch_cr_suite_evals.sh`, `launch_longctx_task_evals.sh`.
- Companion GPU task already done: landmark packing + CP (`landmark-packing-cp-task.md`).
