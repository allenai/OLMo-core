"""Inject chunked-document attention into vLLM's FlexAttention backend.

vLLM's FlexAttention backend (`vllm.v1.attention.backends.flex_attention`)
already exposes a `logical_mask_mod` hook that runs in per-request logical
coordinates. We use it to replace the default causal rule with the chunked
rule from `scripts.lib.chunked_attention`, so document tokens only attend to
their own document plus FREE (query/instruction/answer) tokens.

Hook points (monkey-patched at install time):

  1. `GPUModelRunner._build_attention_metadata` — wrapped so the active
     `input_batch` is parked in a thread-local before the metadata builder
     runs. We need access to `input_batch.token_ids_cpu` to derive per-request
     chunk_ids on the fly.
  2. `FlexAttentionMetadataBuilder.build` — wrapped to (a) build a per-request
     `chunk_ids` tensor from token_ids by scanning for <|doc_start|>/<|doc_end|>
     IDs, (b) swap in a chunked mask_mod that consumes those chunk_ids, and
     (c) rebuild the BlockMask with the new mask_mod.

The patch is idempotent: install() can be called multiple times safely. The
patch is a no-op for any model run with FLEX_ATTENTION disabled or before
`set_doc_token_ids()` has been called.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Optional

import numpy as np
import torch

# Module-level config: set by the caller before LLM.generate.
_DOC_START_ID: Optional[int] = None
_DOC_END_ID: Optional[int] = None
_INSTALLED = False

# Sidecar dict of full-weight tensors stripped from a chunked LoRA (typically
# `lm_head.weight` and `embed_tokens.weight`). Set via `set_full_extras` and
# applied to the live model in a `Worker.load_model` post-hook.
_FULL_EXTRAS: dict[str, "torch.Tensor"] = {}

# Per-request chunk-id sentinels (must match scripts.lib.chunked_attention).
_FREE_CHUNK_ID = -1
_PAD_CHUNK_ID = -2

# Debug counters for the patched metadata builder (populated by install()):
#   calls    — total builder invocations
#   applied  — invocations where the chunked mask_mod was installed + BlockMask rebuilt
#   direct   — rebuilds through the efficient _build_block_mask_direct path
#   fallback — rebuilds through create_block_mask (kv page size != kernel block size,
#              e.g. hybrid models); the chunked mask IS still applied on this path.
_DEBUG_STATE: dict = {}


def get_debug_state() -> dict:
    """Return the live builder-patch counters (empty dict before install())."""
    return _DEBUG_STATE


# Thread-local that the patched runner uses to publish the current batch's
# input_batch reference to the patched metadata builder. We need the token IDs
# from `input_batch.token_ids_cpu` to derive chunk IDs each step.
_local = threading.local()

# ---------------------------------------------------------------------------
# Optional per-stage profiling (CHUNK_PROFILE=1).
#
# Chunked eval runs ~20-29x slower than dense (grouping 2k: 28.6s vs 586s;
# contradiction 2k: ~41s vs 1199s), and the metadata builder is invoked on EVERY
# decode step. A CPU microbenchmark showed the chunk_ids rebuild accounts for
# only ~0.4% of runtime at the 2k rung (4.5s per 500 examples vs 1199s total),
# so the cost is elsewhere. This attributes it per stage instead of guessing.
#
# OFF by default: with CHUNK_PROFILE unset both helpers return immediately and
# no CUDA syncs are inserted, so already-validated numbers are bit-unaffected.
# ---------------------------------------------------------------------------
_PROFILE = os.environ.get("CHUNK_PROFILE") == "1"
_prof_state: dict = {}

# ROUND-2 (2026-08-11) CORRECTION TO HOW THESE NUMBERS READ.
#
# `_prof_add(..., sync=True)` synchronizes BEFORE stopping the clock, so a stage timed that way is
# billed for every GPU kernel that was still in flight when it started -- including the previous
# step's model forward. `build_chunk_ids` was timed with sync=True, which made it look like it cost
# 24.9 ms/call with 8 resident sequences and 0.75 ms/call with 2. A pure-numpy microbenchmark of the
# identical function (debug/chunked_eval_speedup/bench_chunk_ids_r2.py) measures 4.65 ms at 8 reqs
# and 1.18 ms at 2, scaling perfectly linearly. So ~20 of those 24.9 ms were the absorbed GPU tail.
#
# The fix is to drain the queue ONCE at the top of the step under its own label (`sync_head`, the
# previous step's GPU time) and then time the CPU stages without syncing. Every stage below is
# labelled `<stage>.p` on a prefill step and `<stage>.d` on a decode step, because the two have
# completely different cost structures and averaging them hides the answer.
_PROFILE_SPLIT = os.environ.get("CHUNK_PROFILE_SPLIT") == "1"


def _prof_now():
    return time.perf_counter() if _PROFILE else None


def _prof_add(label: str, t0, sync: bool = False) -> None:
    """Accumulate elapsed seconds under `label`.

    `sync=True` forces a CUDA synchronize first, so GPU work *launched* inside
    the stage is attributed to that stage rather than to whatever later call
    happens to block on it. Only done when profiling is enabled.
    """
    if not _PROFILE or t0 is None:
        return
    if sync:
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
    slot = _prof_state.setdefault(label, [0.0, 0])
    slot[0] += time.perf_counter() - t0
    slot[1] += 1


def profile_report() -> dict:
    """Return (and print) {stage: {seconds, calls, ms_per_call}}."""
    out = {}
    for label, (secs, n) in sorted(_prof_state.items(), key=lambda kv: -kv[1][0]):
        out[label] = {"seconds": secs, "calls": n,
                      "ms_per_call": (secs / n * 1000.0) if n else 0.0}
    if _PROFILE:
        print("\n[vllm_chunked_patch] === stage profile ===")
        print(f"{'stage':24s} {'seconds':>10s} {'calls':>9s} {'ms/call':>10s}")
        for label, d in out.items():
            print(f"{label:24s} {d['seconds']:10.2f} {d['calls']:9d} {d['ms_per_call']:10.3f}")
    return out


if _PROFILE:
    # Self-contained: emit the breakdown at interpreter exit so no shared,
    # already-validated driver script has to be edited to get the numbers.
    import atexit

    atexit.register(profile_report)



def set_doc_token_ids(doc_start_id: int, doc_end_id: int) -> None:
    """Configure which token IDs delimit document boundaries.

    Must be called before any `LLM.generate(...)`. Re-callable across runs;
    each call overrides the previous IDs.
    """
    global _DOC_START_ID, _DOC_END_ID
    _DOC_START_ID = int(doc_start_id)
    _DOC_END_ID = int(doc_end_id)


def set_full_extras(tensors: dict) -> None:
    """Register full-weight tensors (e.g. `lm_head.weight`,
    `embed_tokens.weight`) to be installed onto the live vLLM model after
    base-weight load completes.

    Keys are PEFT-format adapter names (containing `lm_head.weight` or
    `embed_tokens.weight`); the post-load hook locates the matching live
    parameter by suffix and slice-assigns the source tensor onto its first
    rows.
    """
    global _FULL_EXTRAS
    _FULL_EXTRAS = dict(tensors)


def install() -> None:
    """Monkey-patch vLLM's FlexAttention backend + GPU runner.

    Idempotent. Safe to call before configuring doc token IDs — the patches
    no-op until `set_doc_token_ids()` is called.
    """
    global _INSTALLED
    if _INSTALLED:
        return
    _patch_runner_build_attention_metadata()
    _patch_flex_metadata_builder()
    _patch_runner_load_model_for_extras()
    _patch_flex_impl_forward_reshape()
    _patch_flex_kernel_options_pow2()
    _INSTALLED = True


# ---------------------------------------------------------------------------
# Runner patch: capture input_batch in a thread-local around metadata build.
# ---------------------------------------------------------------------------

def _patch_runner_build_attention_metadata() -> None:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    orig = GPUModelRunner._build_attention_metadata

    def wrapped(self, *args, **kwargs):
        prev = getattr(_local, "input_batch", None)
        _local.input_batch = self.input_batch
        try:
            return orig(self, *args, **kwargs)
        finally:
            _local.input_batch = prev

    GPUModelRunner._build_attention_metadata = wrapped


# ---------------------------------------------------------------------------
# FlexAttention impl patch: vLLM's stock forward() uses `.view(-1, ...)` on
# the unbound key/value cache, which fails for hybrid (mamba+softmax) Qwen3.5
# because vLLM pads the attention page size by ~5% to align with the mamba
# page size — that padding leaves the cache tensor non-contiguous in the
# expected layout. Swap `.view` for `.reshape` to fall back to a copy when
# strides don't allow a zero-copy view.
# ---------------------------------------------------------------------------

def _patch_flex_kernel_options_pow2() -> None:
    """Force BLOCK_N to a power of 2 in vLLM's FlexAttention kernel options.

    For pure-softmax models, vLLM sets `kv_block_size` to a power of 2 (the
    KV cache block size, e.g. 16 or 128). For hybrid models (Qwen3.5: mamba
    + attention), vLLM pads the page to 288 to align with the mamba page
    size. The triton FlexAttention kernel uses `tl.arange(0, BLOCK_N)`,
    which compiles only when BLOCK_N is a power of 2 — so 288 trips the
    "arange's range must be a power of 2" error during kernel JIT. Clamp
    BLOCK_N to the largest power of 2 that divides the metadata block size
    (so the cache layout still works); 288 -> 32, 16 -> 16, etc.
    """
    import vllm.v1.attention.backends.flex_attention as flex_mod

    orig = flex_mod.get_kernel_options

    def _largest_pow2_divisor(n: int) -> int:
        # bit twiddle: clear all but lowest set bit
        if n <= 0:
            return 1
        return n & -n

    def patched(query, block_m, block_n, use_direct_build: bool):
        block_n_pow2 = _largest_pow2_divisor(int(block_n))
        if block_n_pow2 < block_n:
            # Cap the kernel's inner block to the largest power-of-2 that
            # still divides the logical KV page (288 -> 32).
            block_n = block_n_pow2
        # CHUNK_BLOCK_N: experimental override to measure how much the forced
        # BLOCK_N=32 costs. Profiling (48 ex, contradiction 2k) attributed 83%
        # of chunked runtime to the kernel/forward and only 17% to this patch's
        # metadata work, so the KV tile size is the main remaining suspect:
        # Qwen3.5 pads the attention page to 288 to match the mamba page, and
        # 288 = 2^5 * 9, so the largest power-of-2 divisor is only 32 (vs 128+
        # for a pure-softmax model). Unset -> stock behaviour, unchanged.
        _override = os.environ.get("CHUNK_BLOCK_N")
        if _override:
            block_n = int(_override)
        return orig(query, block_m, block_n, use_direct_build)

    flex_mod.get_kernel_options = patched


def _patch_flex_impl_forward_reshape() -> None:
    from vllm.v1.attention.backends.flex_attention import FlexAttentionImpl

    orig_forward = FlexAttentionImpl.forward

    def patched_forward(self, layer, query, key, value, kv_cache, attn_metadata,
                        output=None, output_scale=None, output_block_scale=None):
        # Mirror the upstream forward but call `.reshape` where it calls
        # `.view`. We reproduce only the decode branch's KV-cache reshape
        # (the cause of the strides crash); everything else delegates to the
        # original implementation by way of a small monkey-patch on the
        # bound tensors.
        #
        # Upstream stock code (vllm/v1/attention/backends/flex_attention.py):
        #     key_cache = key_cache.view(-1, self.num_kv_heads, self.head_size)
        #     value_cache = value_cache.view(-1, self.num_kv_heads, self.head_size)
        #
        # We monkey-patch torch.Tensor.view on the kv_cache pieces by handing
        # vLLM a pre-reshaped view via a tiny shim. Easier: subclass-wrap
        # kv_cache so .unbind(0) returns reshape-friendly tensors. Even
        # easier: replace the whole forward with a copy that uses reshape.
        # We pick the simplest path — hand the original a re-laid-out copy of
        # kv_cache such that `kv_cache.unbind(1)` yields CONTIGUOUS K and V
        # slices (so the subsequent `.view(-1, H, D)` succeeds). A plain
        # `.contiguous()` is NOT enough: for a contiguous
        # (num_blocks, 2, page, H, D) tensor, unbind(1) slices still have a
        # stride jump across the block dim and cannot be flattened by view.
        # transpose(0,1).contiguous() packs K and V each into one contiguous
        # region; transposing back restores the expected indexing. Safe
        # because this forward only READS the cache (writes happen upstream),
        # and cheap (~tens of MB per call at validation cache sizes).
        # VARLEN PREFILL dispatch: steps whose metadata carries a varlen plan never touch
        # the flex kernel or the cache re-layout below (the plan's decode part reads the
        # paged cache directly through flash_attn's block_table support).
        plan = getattr(attn_metadata, "_chunk_varlen_plan", None) \
            if attn_metadata is not None else None
        if plan is not None:
            return _varlen_forward(
                self, layer, query, key, value, kv_cache, attn_metadata, output, plan,
            )
        if (
            attn_metadata is not None
            and kv_cache.numel() > 0
            and kv_cache.dim() >= 2
            and kv_cache.shape[1] == 2
        ):
            kv_cache = kv_cache.transpose(0, 1).contiguous().transpose(0, 1)
        return orig_forward(
            self, layer, query, key, value, kv_cache, attn_metadata,
            output=output, output_scale=output_scale,
            output_block_scale=output_block_scale,
        )

    FlexAttentionImpl.forward = patched_forward


# ---------------------------------------------------------------------------
# Worker patch: paste full-weight tensors from a stripped LoRA onto the live
# model after vLLM finishes loading the base.
# ---------------------------------------------------------------------------

def _patch_runner_load_model_for_extras() -> None:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    orig = GPUModelRunner.load_model

    def wrapped(self, *args, **kwargs):
        ret = orig(self, *args, **kwargs)
        if not _FULL_EXTRAS:
            return ret
        _apply_full_extras_to_model(self.model, _FULL_EXTRAS)
        return ret

    GPUModelRunner.load_model = wrapped


def _apply_full_extras_to_model(model, full_extras: dict) -> None:
    """Slice-copy each `full_extras` tensor onto the matching live parameter.

    Match logic: for an adapter key like `base_model.model.lm_head.weight`,
    find a model parameter whose qualified name ends in `lm_head.weight`. We
    intentionally tolerate vLLM's prefix-y naming (`language_model.lm_head…`)
    and any leading PEFT/multimodal wrappers.

    The source tensor's row-count (e.g. 248079) is typically smaller than the
    target's (the base's padded vocab, e.g. 248320). We copy into rows
    `[:src_rows]` and leave the rest untouched.
    """
    target_keys = ["lm_head.weight", "embed_tokens.weight"]
    by_target: dict[str, "torch.Tensor"] = {}
    for k, t in full_extras.items():
        for tk in target_keys:
            if k.endswith(tk):
                by_target.setdefault(tk, t)
                break

    if not by_target:
        return

    name_to_param = dict(model.named_parameters())
    matched = 0
    for tk, src in by_target.items():
        candidates = [
            (name, p) for name, p in name_to_param.items() if name.endswith(tk)
        ]
        if not candidates:
            print(f"[vllm_chunked_patch] no live param ending in {tk!r}; "
                  f"skipping")
            continue
        for name, p in candidates:
            src_rows = src.shape[0]
            if p.shape[1] != src.shape[1]:
                print(f"[vllm_chunked_patch] skip {name}: hidden-dim mismatch "
                      f"{tuple(p.shape)} vs {tuple(src.shape)}")
                continue
            tgt_rows = p.shape[0]
            if src_rows > tgt_rows:
                print(f"[vllm_chunked_patch] skip {name}: src has more rows "
                      f"({src_rows}) than target ({tgt_rows})")
                continue
            with torch.no_grad():
                p.data[:src_rows].copy_(src.to(dtype=p.dtype, device=p.device))
            matched += 1
            print(f"[vllm_chunked_patch] installed trained {tk} into {name} "
                  f"(rows[:{src_rows}])")
    if matched == 0:
        print("[vllm_chunked_patch] WARNING: no full-weight tensors installed")


# ---------------------------------------------------------------------------
# Builder patch: swap in a chunked mask_mod and rebuild block_mask.
# ---------------------------------------------------------------------------

def _patch_flex_metadata_builder() -> None:
    from vllm.v1.attention.backends.flex_attention import (
        FlexAttentionMetadataBuilder,
    )

    orig_build = FlexAttentionMetadataBuilder.build

    global _DEBUG_STATE
    _DEBUG_STATE = {"calls": 0, "applied": 0, "direct": 0, "fallback": 0, "freepath": 0,
                    "logged": False}
    _debug_state = _DEBUG_STATE

    def wrapped(self, common_prefix_len, common_attn_metadata, fast_build=False):
        # Drain the previous step's GPU work under its own label so the CPU stages below are
        # timed for the CPU work they actually do (see the ROUND-2 note at _PROFILE_SPLIT).
        if _PROFILE_SPLIT:
            _t_sync = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _prof_add("sync_head", _t_sync)
        # A step is a prefill step iff it carries more query tokens than requests.
        _sfx = ""
        if _PROFILE_SPLIT:
            _nq = int(common_attn_metadata.num_actual_tokens)
            _nr = int(common_attn_metadata.num_reqs)
            _sfx = ".p" if _nq > _nr else ".d"

        _t_orig = _prof_now()
        metadata = orig_build(
            self, common_prefix_len, common_attn_metadata, fast_build=fast_build
        )
        _prof_add("orig_build" + _sfx, _t_orig)
        _debug_state["calls"] += 1
        if _DOC_START_ID is None or _DOC_END_ID is None:
            if not _debug_state["logged"]:
                print(f"[vllm_chunked_patch] WARNING: builder hit "
                      f"#{_debug_state['calls']} but doc IDs not set "
                      f"(_DOC_START_ID={_DOC_START_ID}, "
                      f"_DOC_END_ID={_DOC_END_ID})")
                _debug_state["logged"] = True
            return metadata
        ib = getattr(_local, "input_batch", None)
        if ib is None:
            if not _debug_state["logged"]:
                print(f"[vllm_chunked_patch] WARNING: builder hit "
                      f"#{_debug_state['calls']} but no input_batch in "
                      f"thread-local (runner patch not firing)")
                _debug_state["logged"] = True
            return metadata

        # VARLEN PREFILL: if every request in this step is a one-shot full prefill or an
        # all-FREE-query continuation, hand the whole step to the varlen forward — no
        # chunked mask_mod, no BlockMask rebuild, no flex kernel (see the section note at
        # _VARLEN_ENABLED). Falls through to the historical path when the step is mixed.
        if _VARLEN_ENABLED:
            _t_vp = _prof_now()
            plan = _try_build_varlen_plan(ib, common_attn_metadata, metadata)
            _prof_add("varlen_plan" + _sfx, _t_vp)
            if plan is not None:
                metadata._chunk_varlen_plan = plan
                _debug_state["varlen"] = _debug_state.get("varlen", 0) + 1
                if _debug_state["varlen"] <= 3:
                    print(f"[vllm_chunked_patch] varlen step #{_debug_state['calls']}: "
                          f"prefill_reqs={plan['n_pf']} decode_reqs={plan['n_dec']}")
                return metadata

        _t_cid = _prof_now()
        chunk_ids = _build_chunk_ids_for_batch(
            ib, common_attn_metadata, metadata.block_table.device,
        )
        _prof_add("build_chunk_ids" + _sfx, _t_cid, sync=not _PROFILE_SPLIT)
        if chunk_ids is None:
            if _FREEPATH_ENABLED and _LAST_ALL_QUERIES_FREE:
                # Every query token on this step is FREE, so the chunked rule is identical to
                # plain causal (see the FREE-QUERY FAST PATH note). vLLM's own build() already
                # left a correct paged-causal block_mask on `metadata`; leave it alone.
                _debug_state["freepath"] = _debug_state.get("freepath", 0) + 1
                _prof_add("freepath" + _sfx, _t_cid)
                return metadata
            return metadata
        _debug_state["applied"] += 1
        if _debug_state["applied"] <= 3:
            n_doc = int(((chunk_ids >= 0).sum()).item())
            n_free = int(((chunk_ids == -1).sum()).item())
            n_pad = int(((chunk_ids == -2).sum()).item())
            print(f"[vllm_chunked_patch] applied chunked mask call "
                  f"#{_debug_state['calls']}: chunk_ids shape={tuple(chunk_ids.shape)} "
                  f"(doc={n_doc}, free={n_free}, pad={n_pad})")

        # Stash for debugging / inspection on the metadata object.
        metadata._chunk_ids = chunk_ids

        # GUARDRAIL: FlexAttention's kv_block_size MUST divide the KV-cache page size
        # (block_size). If it does not, the flex kernel silently produces WRONG (finite,
        # non-NaN) attention output — every generation degenerates to token-0 "!!!!". This
        # bit us on the Qwen3.5-4B GDN-hybrid, whose attention page is 528 (=16*33), not the
        # 544 an earlier comment assumed: kv_block_size=32 does not divide 528. Fail loudly
        # here (with the divisor to use) rather than emit garbage that reads as a bad model.
        if not _debug_state.get("divchecked"):
            _debug_state["divchecked"] = True
            bs, kvb = int(metadata.block_size), int(metadata.kv_block_size)
            # ROUND-2 ESCAPE HATCH (CHUNK_ALLOW_KVBLOCK_MISMATCH=1), for INVESTIGATION ONLY.
            # The guardrail below is right for the DIRECT-build path, where kv blocks are pages
            # 1:1. On the FALLBACK path -- the only one chunked mode ever takes -- kv_block_size is
            # merely the BlockMask's granularity over flat physical KV indices: create_block_mask
            # marks a block FULL only when mask_mod holds for every element of it, and
            # flex_attention re-applies mask_mod elementwise inside every PARTIAL block. That
            # argument says a non-dividing kv_block_size should be exact, and it is what a larger
            # triton BLOCK_N (get_kernel_options, flex_attention.py:1396) would need. It is NOT
            # confirmed: the historical "!!!!" corruption with kv_block_size=32 is unexplained
            # under that reading. Anything measured through this flag must be validated against a
            # known-good f1 AND parse_rate on a full rung before it goes anywhere near production.
            if bs % kvb != 0 and os.environ.get("CHUNK_ALLOW_KVBLOCK_MISMATCH") == "1":
                print(f"[vllm_chunked_patch] *** CHUNK_ALLOW_KVBLOCK_MISMATCH=1: proceeding with "
                      f"kv_block_size={kvb} which does NOT divide block_size={bs}. UNVALIDATED -- "
                      f"check parse_rate and f1 against a known-good baseline. ***", flush=True)
            elif bs % kvb != 0:
                good = bs & -bs  # largest power-of-2 divisor of block_size
                raise ValueError(
                    f"[vllm_chunked_patch] flex_attn_kv_block_size={kvb} does not divide "
                    f"the KV-cache page size block_size={bs}; this SILENTLY corrupts "
                    f"FlexAttention output. Set flex_attn_kv_block_size={good} (largest "
                    f"power-of-2 dividing {bs}) in the LLM(attention_config=...) call."
                )

        # Replace mask_mod with a chunked variant. The chunked rule uses the
        # request index (which we get from `metadata.doc_ids[q_idx]`) to look
        # up per-request chunk IDs. The default `get_causal_mask_mod` strips
        # the request index when it calls `logical_mask_mod`, so we install
        # our own `final_mask_mod` directly.
        _t_mm = _prof_now()
        metadata.mask_mod = _build_chunked_final_mask_mod(metadata, chunk_ids)
        _prof_add("make_mask_mod" + _sfx, _t_mm)

        # Rebuild block_mask with the new mask_mod. FlexAttention has no
        # update_block_table path (supports_update_block_table=False), so the
        # builder is called every step — same path as the default backend.
        _t_bm = _prof_now()
        if metadata.direct_build and metadata.causal:
            # ⚠ CORRECTNESS LANDMINE (made explicit 2026-08-11). `_build_block_mask_direct()`
            # constructs the BlockMask from the page table + causal structure ALONE; it never
            # evaluates `mask_mod`. Taking this branch therefore SILENTLY DISCARDS the chunked
            # mask and produces dense-causal numbers wearing a "chunked" label -- the worst
            # possible failure, because it looks like a modelling result.
            #
            # It is unreachable today only by accident: vLLM sets `direct_build=False` whenever
            # `kv_block_size != block_size` (flex_attention.py:882), and the Qwen3.5 GDN-hybrid's
            # attention page is 528 while kv_block_size must be a power of two. Any future change
            # that makes those equal would re-arm it. Fail loudly instead of falling through.
            _debug_state["direct"] += 1
            raise RuntimeError(
                "[vllm_chunked_patch] direct_build=True: vLLM's _build_block_mask_direct() "
                "ignores mask_mod, so the CHUNKED MASK WOULD BE SILENTLY DROPPED and this run "
                "would report dense numbers as chunked. This happens when "
                f"kv_block_size ({metadata.kv_block_size}) == KV page block_size "
                f"({metadata.block_size}). Set flex_attn_kv_block_size to a power of two "
                "strictly smaller than the page size to stay on the fallback path."
            )
        else:
            # ⚠ metadata.build_block_mask() recomputes `self.get_mask_mod()`
            # internally — the DEFAULT causal mask_mod — and would silently
            # DROP the chunked mask_mod we just installed (hybrid models hit
            # this path whenever kv page size != kernel block size, i.e.
            # direct_build=False). Build the BlockMask ourselves from
            # `metadata.mask_mod` (the chunked one) instead — this is
            # build_block_mask() verbatim with the mask_mod swapped.
            _debug_state["fallback"] += 1
            from vllm.v1.attention.backends.flex_attention import (
                create_block_mask_compiled,
            )
            kv_len = (
                metadata.total_cache_tokens
                if metadata.uses_paged_kv
                else metadata.num_actual_tokens
            )
            metadata.block_mask = create_block_mask_compiled(
                metadata.mask_mod,
                None,
                None,
                metadata.num_actual_tokens,
                kv_len,
                device=metadata.block_table.device,
                BLOCK_SIZE=(metadata.q_block_size, metadata.kv_block_size),
            )
        _prof_add("build_block_mask" + _sfx, _t_bm, sync=True)
        return metadata

    FlexAttentionMetadataBuilder.build = wrapped


# ---------------------------------------------------------------------------
# Chunk-id derivation from token IDs.
# ---------------------------------------------------------------------------

# ROUND-2 optimization (CHUNK_CACHE_IDS=1, OFF by default).
#
# The row of chunk ids for a request is recomputed from scratch on EVERY forward step, even though
# during generation the only thing that changed is that the sequence grew by one token. The scan is
# ~4.65 ms per step with 8 resident 8.8k-token sequences (bench_chunk_ids_r2.py), ~6% of an 8k-rung
# run. Cache the row per request and extend it.
#
# EXACTNESS. The extension is only taken when the newly-appended tokens contain NO doc_start and NO
# doc_end. Under that condition an extended row is elementwise identical to a full recompute:
#   * positions < prev_len -- chunk ids come only from (start, end) pairs lying entirely below
#     prev_len, and the set of markers below prev_len has not changed, so every assignment (and
#     every deliberately-unassigned trailing unmatched start) is reproduced;
#   * positions >= prev_len -- contain no marker, so a full recompute would leave them FREE, which
#     is what the extension writes.
# If a marker DOES appear in the new tokens (a model that generates <|doc_start|>), the cache falls
# back to the full recompute, so the output is unconditionally identical to the uncached path.
_CID_CACHE: dict = {}
_CID_ENABLED = os.environ.get("CHUNK_CACHE_IDS") == "1"

# ---------------------------------------------------------------------------
# FREE-QUERY FAST PATH (CHUNK_FREE_QUERY_FASTPATH=1, OFF by default).
#
# The chunked rule (document_chunked.py:9) is
#     allowed = causal & not_pad & (context_ok | q_free | kv_free)
# so when the QUERY token is FREE, `q_free` short-circuits the parenthesis to True and the rule
# degenerates to `causal & not_pad` -- i.e. EXACTLY vLLM's default paged-causal mask. The chunked
# mask constrains context->context attention only; it never constrains a FREE query.
#
# Every token generated during decoding is FREE (it is answer text, not a document). So on any step
# whose query tokens are all FREE -- which is 93.7% of steps in the 8k production run -- installing
# the chunked mask_mod and rebuilding the BlockMask is pure overhead that computes, at great
# expense, the mask vLLM already built for free in `build()` (flex_attention.py:1141).
#
# This flag detects that case and returns vLLM's own metadata untouched. It is not an
# approximation: the condition is checked per step against the actual chunk ids of the actual query
# positions, so a step containing even one context query takes the normal chunked path. If the
# model ever emitted a <|doc_start|> mid-answer, those positions would stop being FREE and the
# check would fail closed.
_FREEPATH_ENABLED = os.environ.get("CHUNK_FREE_QUERY_FASTPATH") == "1"
_LAST_ALL_QUERIES_FREE = False


def _query_positions_all_free(chunk_rows, seq_lens, common_attn_metadata, num_reqs) -> bool:
    """True iff every QUERY position in this step has chunk id FREE.

    Query tokens for request i occupy logical positions [seq_len_i - q_len_i, seq_len_i).
    `chunk_rows` is the (num_reqs, max_len) int32 numpy array of chunk ids.
    """
    qsl = getattr(common_attn_metadata, "query_start_loc_cpu", None)
    if qsl is None:
        qsl = common_attn_metadata.query_start_loc.cpu()
    qsl = qsl.numpy()
    for ri in range(num_reqs):
        q_len = int(qsl[ri + 1] - qsl[ri])
        if q_len <= 0:
            continue
        slen = int(seq_lens[ri])
        lo = slen - q_len
        if lo < 0:
            return False
        if not np.all(chunk_rows[ri, lo:slen] == _FREE_CHUNK_ID):
            return False
    return True


def _build_chunk_ids_row(ids: np.ndarray) -> np.ndarray:
    """Chunk-id row for one request's `ids` (length = seq_len). Extracted verbatim from the
    original per-request body of `_build_chunk_ids_for_batch` so both paths share one rule."""
    row = np.full(ids.shape[0], _FREE_CHUNK_ID, dtype=np.int32)
    starts = np.flatnonzero(ids == _DOC_START_ID)
    ends = np.flatnonzero(ids == _DOC_END_ID)
    if starts.size == 0 or ends.size == 0:
        return row
    ei = 0
    chunk_idx = 0
    for s in starts:
        while ei < ends.size and ends[ei] < s:
            ei += 1
        if ei >= ends.size:
            break
        e = ends[ei]
        row[s : e + 1] = chunk_idx
        chunk_idx += 1
        ei += 1
    return row


def _build_chunk_ids_for_batch(
    input_batch, common_attn_metadata, device: torch.device,
) -> Optional[torch.Tensor]:
    """Build a (num_reqs, max_seq_len) int32 tensor of chunk IDs.

    Scans `input_batch.token_ids_cpu[req, :seq_len]` for matching
    <|doc_start|>...<|doc_end|> pairs, assigning chunk indices 0, 1, 2, ...
    Tokens outside any chunk get FREE_CHUNK_ID (-1); positions past seq_len
    get PAD_CHUNK_ID (-2) but are never accessed because `is_valid` in the
    final_mask_mod gates by seq_len.

    Returns None if there are zero requests in the batch.
    """
    num_reqs = common_attn_metadata.num_reqs
    if num_reqs == 0:
        return None

    seq_lens = common_attn_metadata.seq_lens.cpu().numpy()[:num_reqs].astype(np.int64)
    max_len = int(seq_lens.max()) if num_reqs > 0 else 0
    if max_len == 0:
        return None

    chunk_ids = np.full((num_reqs, max_len), _PAD_CHUNK_ID, dtype=np.int32)
    token_ids_cpu = input_batch.token_ids_cpu  # (max_num_reqs, max_num_tokens) numpy
    req_ids = input_batch.req_ids if _CID_ENABLED else None
    for ri in range(num_reqs):
        slen = int(seq_lens[ri])
        if slen <= 0:
            continue
        ids = token_ids_cpu[ri, :slen]

        if _CID_ENABLED:
            key = req_ids[ri]
            cached = _CID_CACHE.get(key)
            if cached is not None:
                prev_len, prev_row = cached
                if 0 < prev_len <= slen:
                    tail = ids[prev_len:slen]
                    if not (
                        np.any(tail == _DOC_START_ID) or np.any(tail == _DOC_END_ID)
                    ):
                        # Extension is provably identical to a full recompute (see note above).
                        row = np.full(slen, _FREE_CHUNK_ID, dtype=np.int32)
                        row[:prev_len] = prev_row
                        chunk_ids[ri, :slen] = row
                        _CID_CACHE[key] = (slen, row)
                        continue
            row = _build_chunk_ids_row(ids)
            chunk_ids[ri, :slen] = row
            _CID_CACHE[key] = (slen, row)
            continue

        # Default everything in the live region to FREE; doc spans get filled in.
        chunk_ids[ri, :slen] = _FREE_CHUNK_ID

        starts = np.flatnonzero(ids == _DOC_START_ID)
        ends = np.flatnonzero(ids == _DOC_END_ID)
        if starts.size == 0 or ends.size == 0:
            continue
        # Match each <|doc_start|> with the next <|doc_end|>. We don't assume
        # they're perfectly interleaved (a partial-prefill request may have a
        # doc_start with no matching doc_end yet); we just stop on the first
        # unmatched start.
        ei = 0
        chunk_idx = 0
        for s in starts:
            while ei < ends.size and ends[ei] < s:
                ei += 1
            if ei >= ends.size:
                break
            e = ends[ei]
            chunk_ids[ri, s : e + 1] = chunk_idx
            chunk_idx += 1
            ei += 1

    if _CID_ENABLED and len(_CID_CACHE) > 4 * max(num_reqs, 1):
        live = set(req_ids[:num_reqs])
        for k in [k for k in _CID_CACHE if k not in live]:
            del _CID_CACHE[k]

    global _LAST_ALL_QUERIES_FREE
    if _FREEPATH_ENABLED:
        _LAST_ALL_QUERIES_FREE = _query_positions_all_free(
            chunk_ids, seq_lens, common_attn_metadata, num_reqs
        )
        if _LAST_ALL_QUERIES_FREE:
            # Caller will discard this; skip the H2D copy too.
            return None

    return torch.from_numpy(chunk_ids).to(device=device, non_blocking=True)


# ---------------------------------------------------------------------------
# VARLEN PREFILL (CHUNK_VARLEN_PREFILL=1, OFF by default). ROUND 5, 2026-08-13.
#
# The chunked rule on a pure full-prompt prefill decomposes EXACTLY into three
# pieces, none of which needs a custom mask (proof + float64 test:
# debug/chunked_eval_speedup/test_varlen_decomposition.py):
#
#   A. causal attention within each maximal constant-chunk-id run (documents
#      AND free runs alike)                        -> flash_attn_varlen_func
#   B. FREE query -> every token strictly before its own run's start (full)
#   C. doc  query -> every FREE token strictly before it (full)
#
# Each allowed (q, kv) pair is covered by exactly one of A/B/C, so merging the
# (out, lse) of A with the disjoint (B|C) via online softmax (vLLM's
# merge_attn_states) reproduces full-rule attention. B and C are thin strips
# (|FREE| is a few hundred tokens), computed as batched fp32 matmuls.
#
# Decode is handled in the same step: a generated token is FREE, so its rule
# is plain causal over its whole sequence — exactly what paged
# flash_attn_varlen_func(block_table=..., causal=True) computes natively.
#
# Net effect on an eligible step: NO chunked BlockMask is ever built, the flex
# kernel never runs, and the whole-KV-cache transpose copy in
# _patch_flex_impl_forward_reshape is skipped. A step qualifies when every
# request in it is either (a) a full one-shot prefill (q_len == seq_len; pair
# with CHUNK_MAX_BATCHED_TOKENS >= longest prompt) or (b) a continuation whose
# query tokens are all FREE. Anything else (chunked/partial prefill, a model
# that emitted doc markers mid-generation) falls back to the flex path for
# that step, which stays bit-identical to historical behavior.
# ---------------------------------------------------------------------------
_VARLEN_ENABLED = os.environ.get("CHUNK_VARLEN_PREFILL") == "1"
# Per-request tail tracker: req_id -> tokens scanned so far. A continuation's
# query tokens are provably FREE if no doc marker has appeared at or after
# min(seen, seq_len - q_len): doc spans need BOTH markers inside the scanned
# row, so a marker-free tail can never be inside a span (same argument as the
# CHUNK_CACHE_IDS exactness note above). Any marker in the tail forces a full
# rescan, so the check fails closed.
_VARLEN_TAIL: dict = {}
_FA_FUNCS: dict = {}
# Diagnostic split (round-C attribution): CHUNK_VARLEN_DECODE=0 makes any step containing
# a continuation request ineligible, so varlen covers PREFILL-ONLY steps and decode stays
# on the historical flex path. Separates prefill-side from decode-side cost.
_VARLEN_DECODE = os.environ.get("CHUNK_VARLEN_DECODE", "1") == "1"


def _get_fa_funcs():
    if not _FA_FUNCS:
        from vllm.v1.attention.ops.merge_attn_states import merge_attn_states
        from vllm.vllm_flash_attn import flash_attn_varlen_func
        _FA_FUNCS["varlen"] = flash_attn_varlen_func
        _FA_FUNCS["merge"] = merge_attn_states
    return _FA_FUNCS


def _try_build_varlen_plan(input_batch, cam, metadata):
    """Classify every request in the step; build the per-step varlen plan.

    Returns None (-> caller falls through to the flex path) unless EVERY
    request is a full one-shot prefill or an all-FREE-query continuation.
    """
    num_reqs = cam.num_reqs
    if num_reqs == 0:
        return None
    # The manual strips implement plain scaled-dot-product only.
    if getattr(metadata, "transformed_score_mod", None) is not None:
        return None
    qsl = cam.query_start_loc_cpu.numpy()
    seq_lens = cam.seq_lens_cpu.numpy()[:num_reqs].astype(np.int64)
    token_ids_cpu = input_batch.token_ids_cpu
    req_ids = input_batch.req_ids

    pf, dec = [], []
    for ri in range(num_reqs):
        q_len = int(qsl[ri + 1] - qsl[ri])
        slen = int(seq_lens[ri])
        if q_len <= 0 or slen <= 0 or q_len > slen:
            return None
        if q_len == slen:
            pf.append(ri)
            continue
        # Continuation: eligible iff all query tokens are FREE.
        ids = token_ids_cpu[ri]
        seen = _VARLEN_TAIL.get(req_ids[ri])
        lo = slen - q_len if seen is None else min(int(seen), slen - q_len)
        tail = ids[lo:slen]
        if np.any(tail == _DOC_START_ID) or np.any(tail == _DOC_END_ID):
            row = _build_chunk_ids_row(ids[:slen])
            if not np.all(row[slen - q_len : slen] == _FREE_CHUNK_ID):
                return None  # doc-token queries mid-continuation: flex path
        if not _VARLEN_DECODE:
            return None
        dec.append(ri)

    device = metadata.block_table.device
    plan: dict = {"n_pf": len(pf), "n_dec": len(dec)}

    if pf:
        sel_np = np.concatenate([np.arange(qsl[ri], qsl[ri + 1]) for ri in pf])
        cu = [0]
        reqs = []
        max_seg = 1
        base = 0
        for ri in pf:
            slen = int(seq_lens[ri])
            row = _build_chunk_ids_row(token_ids_cpu[ri, :slen])
            cuts = np.flatnonzero(row[1:] != row[:-1]) + 1
            bounds = np.concatenate(([0], cuts, [slen]))
            seg_lens = np.diff(bounds)
            max_seg = max(max_seg, int(seg_lens.max()))
            cu.extend((base + bounds[1:]).tolist())
            free = row == _FREE_CHUNK_ID
            free_idx = np.flatnonzero(free)
            doc_idx = np.flatnonzero(~free)
            run_start = bounds[:-1][np.repeat(np.arange(seg_lens.size), seg_lens)]
            reqs.append(dict(
                base=base,
                L=slen,
                free_idx=torch.from_numpy(free_idx).to(device),
                free_cut=torch.from_numpy(run_start[free_idx].copy()).to(device),
                doc_idx=torch.from_numpy(doc_idx).to(device),
            ))
            base += slen
        plan["pf_sel"] = torch.from_numpy(sel_np).to(device)
        plan["pf_cu"] = torch.tensor(cu, dtype=torch.int32, device=device)
        plan["pf_max_seg"] = max_seg
        plan["pf_reqs"] = reqs

    if dec:
        d_sel = np.concatenate([np.arange(qsl[ri], qsl[ri + 1]) for ri in dec])
        d_qlens = np.array([int(qsl[ri + 1] - qsl[ri]) for ri in dec], dtype=np.int64)
        cu_q = np.zeros(len(dec) + 1, dtype=np.int32)
        np.cumsum(d_qlens, out=cu_q[1:])
        dec_t = torch.tensor(dec, dtype=torch.long, device=device)
        plan["dec_sel"] = torch.from_numpy(d_sel).to(device)
        plan["dec_cu_q"] = torch.from_numpy(cu_q).to(device)
        plan["dec_max_q"] = int(d_qlens.max())
        plan["dec_block_table"] = metadata.block_table[dec_t]
        plan["dec_seqused"] = cam.seq_lens.to(device)[dec_t].to(torch.int32)
        plan["dec_max_k"] = int(seq_lens[dec].max())

    # Advance the tail tracker for every request in the step, and evict dead ids.
    for ri in pf + dec:
        _VARLEN_TAIL[req_ids[ri]] = int(seq_lens[ri])
    if len(_VARLEN_TAIL) > 4 * max(num_reqs, 1):
        live = set(req_ids[:num_reqs])
        for k in [k for k in _VARLEN_TAIL if k not in live]:
            del _VARLEN_TAIL[k]

    return plan


def _strip_attn(q, k, v, mask, scale):
    """Masked attention for the thin B/C strips, grouped-GQA, fp32.

    q: (m, Hq, D); k, v: (n, Hkv, D); mask: (m, n) bool, True = allowed.
    Returns out (m, Hq, D) in q.dtype and lse (Hq, m) fp32. Rows with no
    allowed kv get out = 0 and lse = -inf (merge then keeps the other part).
    """
    m, hq, d = q.shape
    n, hkv, _ = k.shape
    g = hq // hkv
    qf = q.float().view(m, hkv, g, d)
    scores = torch.einsum("mkgd,nkd->kgmn", qf, k.float()) * scale
    scores = scores.masked_fill(~mask[None, None], float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)  # (hkv, g, m); -inf on all-masked rows
    attn = torch.softmax(scores, dim=-1).nan_to_num_(0.0)
    out = torch.einsum("kgmn,nkd->mkgd", attn, v.float()).reshape(m, hq, d)
    return out.to(q.dtype), lse.reshape(hq, m)


def _varlen_forward(impl, layer, query, key, value, kv_cache, attn_metadata, output, plan):
    """Replace FlexAttentionImpl.forward on a step with a varlen plan.

    The metadata deliberately carries NO chunked BlockMask on these steps, so
    an unsupported layer config cannot fall back silently — it must raise.
    """
    if (
        impl.sliding_window is not None
        or (impl.logits_soft_cap or 0) != 0
        or getattr(layer, "logical_mask_mod", None) is not None
    ):
        raise RuntimeError(
            "[vllm_chunked_patch] CHUNK_VARLEN_PREFILL=1 hit a layer with "
            f"sliding_window={impl.sliding_window} "
            f"logits_soft_cap={impl.logits_soft_cap} or a logical_mask_mod. "
            "The varlen path implements plain scaled-dot-product attention "
            "only; run this model without CHUNK_VARLEN_PREFILL."
        )
    fa = _get_fa_funcs()
    n_tok = attn_metadata.num_actual_tokens
    q = query[:n_tok]
    k = key[:n_tok]
    v = value[:n_tok]
    out_view = output[:n_tok]

    _t_pf = _prof_now()
    if plan["n_pf"]:
        sel = plan["pf_sel"]
        qp, kp, vp = q[sel], k[sel], v[sel]
        out_a, lse_a = fa["varlen"](
            qp, kp, vp,
            plan["pf_max_seg"], plan["pf_cu"], plan["pf_max_seg"],
            cu_seqlens_k=plan["pf_cu"],
            softmax_scale=impl.scale, causal=True, return_softmax_lse=True,
        )
        other = torch.zeros_like(out_a)
        lse_o = torch.full(
            (impl.num_heads, out_a.shape[0]), float("-inf"),
            dtype=torch.float32, device=q.device,
        )
        for r in plan["pf_reqs"]:
            b, seq_l = r["base"], r["L"]
            qr, kr, vr = qp[b : b + seq_l], kp[b : b + seq_l], vp[b : b + seq_l]
            fi, fc, di = r["free_idx"], r["free_cut"], r["doc_idx"]
            if fi.numel() == 0:
                continue  # no FREE tokens: B and C are both empty
            # B: FREE q -> everything strictly before its own run.
            mask_b = torch.arange(seq_l, device=q.device)[None, :] < fc[:, None]
            ob, lb = _strip_attn(qr[fi], kr, vr, mask_b, impl.scale)
            other[b + fi] = ob
            lse_o[:, b + fi] = lb
            if di.numel():
                # C: doc q -> FREE tokens strictly before it.
                mask_c = fi[None, :] < di[:, None]
                oc, lc = _strip_attn(qr[di], kr[fi], vr[fi], mask_c, impl.scale)
                other[b + di] = oc
                lse_o[:, b + di] = lc
        merged = torch.empty_like(out_a)
        fa["merge"](merged, out_a, lse_a, other, lse_o)
        out_view.index_copy_(0, sel, merged.to(out_view.dtype))
    _prof_add("varlen_fwd_pf", _t_pf, sync=True)

    _t_dec = _prof_now()
    if plan["n_dec"]:
        dsel = plan["dec_sel"]
        key_cache, value_cache = kv_cache.unbind(1)
        out_d = fa["varlen"](
            q[dsel], key_cache, value_cache,
            plan["dec_max_q"], plan["dec_cu_q"], plan["dec_max_k"],
            seqused_k=plan["dec_seqused"],
            block_table=plan["dec_block_table"],
            softmax_scale=impl.scale, causal=True,
        )
        out_view.index_copy_(0, dsel, out_d.to(out_view.dtype))
    _prof_add("varlen_fwd_dec", _t_dec, sync=True)

    return output


# ---------------------------------------------------------------------------
# Chunked final mask_mod (replaces FlexAttentionMetadata.mask_mod).
# ---------------------------------------------------------------------------

def _build_chunked_final_mask_mod(metadata, chunk_ids: torch.Tensor):
    """Mirror of `FlexAttentionMetadata.get_causal_mask_mod`, but apply the
    chunked rule on top of causal in logical coordinates.

    The chunked rule (per scripts.lib.chunked_attention):

        attend(q, kv) = causal AND (
            same_chunk(q, kv) OR q_is_free OR kv_is_free
        )

    where chunk_ids[req_idx, logical_position] gives:
        >= 0  → document index (q can only attend to same-doc tokens)
        FREE  → query/instruction/answer (attends to and from everything)
        PAD   → past-end-of-seq (already gated by `is_valid`)
    """
    doc_ids = metadata.doc_ids  # (num_actual_tokens,) int32 — packed q -> request idx
    convert = metadata._convert_physical_to_logical

    def final_mask_mod(b, h, q_idx, physical_kv_idx):
        is_valid, logical_q_idx, logical_kv_idx = convert(
            doc_ids, q_idx, physical_kv_idx
        )
        # Look up which request this query belongs to. doc_ids is the
        # request_lookup — same tensor used in _convert_physical_to_logical.
        req_idx = doc_ids[q_idx]

        # Clamp logical indices into chunk_ids' bounds before gather so we
        # don't OOB on invalid positions; is_valid masks them out below.
        # (The generic create_block_mask path evaluates the FULL physical
        # grid, so garbage block-table slots produce arbitrarily large
        # logical indices — the direct path never sees those, but the
        # fallback does, and an unclamped gather device-asserts.)
        max_pos = chunk_ids.shape[1] - 1
        safe_q = torch.clamp(logical_q_idx, min=0, max=max_pos)
        safe_kv = torch.clamp(logical_kv_idx, min=0, max=max_pos)
        q_chunk = chunk_ids[req_idx, safe_q]
        kv_chunk = chunk_ids[req_idx, safe_kv]

        causal = logical_q_idx >= logical_kv_idx
        same_chunk = (q_chunk == kv_chunk) & (q_chunk >= 0)
        q_free = q_chunk == _FREE_CHUNK_ID
        kv_free = kv_chunk == _FREE_CHUNK_ID
        chunked_ok = same_chunk | q_free | kv_free

        return torch.where(is_valid, causal & chunked_ok, False)

    return final_mask_mod
