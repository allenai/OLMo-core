"""Inject chunked-document attention into vLLM's FlexAttention backend.

vLLM's FlexAttention backend (`vllm.v1.attention.backends.flex_attention`)
already exposes a `logical_mask_mod` hook that runs in per-request logical
coordinates. We use it to replace the default causal rule with the chunked
rule from `ctc_eval.lib.chunked_attention`, so document tokens only attend to
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

import threading
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

# Per-request chunk-id sentinels (must match ctc_eval.lib.chunked_attention).
_FREE_CHUNK_ID = -1
_PAD_CHUNK_ID = -2

# Thread-local that the patched runner uses to publish the current batch's
# input_batch reference to the patched metadata builder. We need the token IDs
# from `input_batch.token_ids_cpu` to derive chunk IDs each step.
_local = threading.local()


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
        # We pick the simplest path — call the original but pre-`contiguous()`
        # the kv_cache. `.contiguous()` is a no-op when strides already match.
        if (
            attn_metadata is not None
            and getattr(attn_metadata, "causal", False)
            and kv_cache.numel() > 0
            and not kv_cache.is_contiguous()
        ):
            kv_cache = kv_cache.contiguous()
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

    _debug_state = {"calls": 0, "applied": 0, "logged": False}

    def wrapped(self, common_prefix_len, common_attn_metadata, fast_build=False):
        metadata = orig_build(
            self, common_prefix_len, common_attn_metadata, fast_build=fast_build
        )
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

        chunk_ids = _build_chunk_ids_for_batch(
            ib, common_attn_metadata, metadata.block_table.device,
        )
        if chunk_ids is None:
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

        # Replace mask_mod with a chunked variant. The chunked rule uses the
        # request index (which we get from `metadata.doc_ids[q_idx]`) to look
        # up per-request chunk IDs. The default `get_causal_mask_mod` strips
        # the request index when it calls `logical_mask_mod`, so we install
        # our own `final_mask_mod` directly.
        metadata.mask_mod = _build_chunked_final_mask_mod(metadata, chunk_ids)

        # Rebuild block_mask with the new mask_mod. FlexAttention has no
        # update_block_table path (supports_update_block_table=False), so the
        # builder is called every step — same path as the default backend.
        if metadata.direct_build and metadata.causal:
            metadata.block_mask = metadata._build_block_mask_direct()
        else:
            metadata.block_mask = metadata.build_block_mask()
        return metadata

    FlexAttentionMetadataBuilder.build = wrapped


# ---------------------------------------------------------------------------
# Chunk-id derivation from token IDs.
# ---------------------------------------------------------------------------

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
    for ri in range(num_reqs):
        slen = int(seq_lens[ri])
        if slen <= 0:
            continue
        ids = token_ids_cpu[ri, :slen]
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

    return torch.from_numpy(chunk_ids).to(device=device, non_blocking=True)


# ---------------------------------------------------------------------------
# Chunked final mask_mod (replaces FlexAttentionMetadata.mask_mod).
# ---------------------------------------------------------------------------

def _build_chunked_final_mask_mod(metadata, chunk_ids: torch.Tensor):
    """Mirror of `FlexAttentionMetadata.get_causal_mask_mod`, but apply the
    chunked rule on top of causal in logical coordinates.

    The chunked rule (per ctc_eval.lib.chunked_attention):

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

        # Clamp logical indices to >= 0 before gather so we don't OOB on
        # invalid positions; is_valid will mask them out below.
        safe_q = torch.clamp(logical_q_idx, min=0)
        safe_kv = torch.clamp(logical_kv_idx, min=0)
        q_chunk = chunk_ids[req_idx, safe_q]
        kv_chunk = chunk_ids[req_idx, safe_kv]

        causal = logical_q_idx >= logical_kv_idx
        same_chunk = (q_chunk == kv_chunk) & (q_chunk >= 0)
        q_free = q_chunk == _FREE_CHUNK_ID
        kv_free = kv_chunk == _FREE_CHUNK_ID
        chunked_ok = same_chunk | q_free | kv_free

        return torch.where(is_valid, causal & chunked_ok, False)

    return final_mask_mod
