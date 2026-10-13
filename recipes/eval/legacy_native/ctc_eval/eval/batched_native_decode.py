"""
Batched greedy decode for the NATIVE olmo_core document-chunked evaluators
(``eval_lc_native_docchunk.py`` / ``eval_lc_native_docchunk_contra.py``).

The bs=1 loop in those files does one example at a time: prefill (chunked mask + KV cache), then a
plain-causal single-token decode loop. That is slow (torch.compile aside, GPU occupancy at bs=1 decode
is poor). This module batches ``--batch-size`` examples per forward, LEFT-padding to the batch's max
prefill length so every row's "next token" always lands in the tensor's last column (no ragged
per-row gather needed) and using the model's existing ``cache_leftpad`` support (already threaded
through :meth:`~olmo_core.nn.transformer.model.Transformer.forward` for exactly this purpose) so the KV
cache and the reconstructed ``chunk_ids`` both correctly exclude the padding.

Greedy decode is deterministic, so a correct implementation must match the bs=1 loop's output
EXACTLY (see the parity harness this ships with, ``debug/batched_eval_parity/``) -- not "close", not
"mostly". Two things had to be handled for exactness, beyond the obvious left-pad + ``cache_leftpad``
attention wiring that :class:`~olmo_core.nn.attention.document_chunked.DocumentChunkedAttention`
already supported:

1. **The "full" (plain-causal, no document mask) arm still needs pad exclusion.** Calling
   ``enable_document_chunk_attention`` installs the model's TRAINED chunked mask -- not what "full"
   is measuring. Instead this module temporarily forces every ``DocumentChunkedAttention`` layer's
   pattern to ``"standard"`` (``causal & not_pad``, no document isolation) for the duration of a
   batched "full" call, via :func:`force_standard_pattern`, then restores it.
2. **Hybrid Qwen3.5 models mix ``DocumentChunkedAttention`` with recurrent ``GatedDeltaNet`` layers**
   (block_pattern e.g. ``["gdn","gdn","gdn","attn"]``). Attention is automatically invariant to
   whatever garbage sits in a left-padded prefix (causal masking + the chunked-mask's own PAD role
   both exclude it). GatedDeltaNet's causal-conv + delta-rule recurrent scan is **not** automatically
   invariant -- a padded row's real tokens would be corrupted by the pad tokens' conv/decay
   contributions unless masked. See :class:`~olmo_core.nn.attention.recurrent.GatedDeltaNet`'s
   ``cache_leftpad`` handling (added alongside this module) for the fix; nothing extra is needed here
   beyond passing ``cache_leftpad`` through, which the model already threads to every block.

Landmark (:class:`~olmo_core.nn.attention.landmark_document.DocumentLandmarkAttention`) is
deliberately NOT supported here -- its periodic landmark-token re-insertion during decode doesn't fit
this loop and batching it correctly is a separate effort. Callers should keep the bs=1 path for
``--variant landmark``.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, List, Optional

import torch

__all__ = [
    "left_pad_batch",
    "force_standard_pattern",
    "model_has_document_chunked_attention",
    "generate_batch_docchunk",
]


def model_has_document_chunked_attention(gm) -> bool:
    """
    Whether any block's sequence mixer is
    :class:`~olmo_core.nn.attention.document_chunked.DocumentChunkedAttention`.

    Some "full" checkpoints in this eval suite are genuinely trained with **plain**
    :class:`~olmo_core.nn.attention.Attention` (a distinct baseline model, not the chunked model
    evaluated without its mask) -- e.g. ``ctc-1ep-contra-full``. Calling
    ``enable_document_chunk_attention`` on such a model would thread a ``chunk_ids`` kwarg into
    blocks whose ``forward`` doesn't accept it (a hard crash), and it isn't needed anyway: plain
    ``Attention``'s KV-cached prefill already goes through ``flash_attn_with_kvcache``, which honors
    ``cache_leftpad`` natively (the same mechanism the generic
    :meth:`~olmo_core.generate.generation_module.transformer.TransformerGenerationModule.generate_batch`
    relies on) -- no chunk-id/pattern setup required. Only checkpoints whose ATTENTION layers really
    are ``DocumentChunkedAttention`` (i.e. "full" means "this chunked model, mask disabled") need
    :func:`force_standard_pattern` and a ``pad_id``-bearing ``enable_document_chunk_attention`` call.
    """
    from olmo_core.nn.attention.document_chunked import DocumentChunkedAttention

    return any(
        isinstance(block.attention, DocumentChunkedAttention)
        for block in gm.model.blocks.values()
    )


def left_pad_batch(
    prefills: List[List[int]], pad_id: int, device: torch.device
) -> "tuple[torch.Tensor, torch.Tensor]":
    """
    LEFT-pad a batch of token-id lists to their common max length.

    :param prefills: One token-id list per example (no padding).
    :param pad_id: The reserved id to fill the pad prefix with (must be marked PAD by the model's
        ``chunk_ids`` reconstruction -- pass it as ``enable_document_chunk_attention``'s ``pad_id``).
    :param device: Target device for the returned tensors.

    :returns: ``(padded, leftpad)`` where ``padded`` is ``(B, W)`` (``W = max(len(p) for p in
        prefills)``, every row's real tokens flush against the right edge) and ``leftpad`` is
        ``(B,)`` int32, the per-row left-pad count (``W - len(prefills[i])``).
    """
    B = len(prefills)
    W = max(len(p) for p in prefills)
    leftpad = torch.tensor([W - len(p) for p in prefills], dtype=torch.int32, device=device)
    padded = torch.full((B, W), pad_id, dtype=torch.long, device=device)
    for i, p in enumerate(prefills):
        if p:
            padded[i, W - len(p) :] = torch.tensor(p, dtype=torch.long, device=device)
    return padded, leftpad


@contextmanager
def force_standard_pattern(gm):
    """
    Temporarily force every :class:`~olmo_core.nn.attention.document_chunked.DocumentChunkedAttention`
    layer in ``gm.model`` to the ``"standard"`` pattern (``causal & not_pad``, i.e. plain causal
    attention that still excludes padding -- no document isolation), restoring each layer's original
    pattern on exit.

    Use this around a batched ``--variant full`` call: the "full" arm means "evaluate this model with
    NO document-chunk restriction", which at bs=1 is achieved by simply never calling
    ``enable_document_chunk_attention`` (so ``chunk_ids`` stays ``None`` and the plain-causal fallback
    in ``DocumentChunkedAttention._sdpa_masked`` kicks in). That fallback has no notion of padding, so
    a left-padded batch would let every row attend the garbage pad prefix. Threading ``chunk_ids``
    through (needed for pad exclusion) would normally apply the model's TRAINED pattern (e.g.
    ``"chunked"``) -- wrong for the "full" arm -- so this swaps every layer's pattern to ``"standard"``
    for the duration instead, which is exactly "causal, pad-excluded, no other restriction".
    """
    from olmo_core.nn.attention.chunked_mask import AttentionPattern
    from olmo_core.nn.attention.document_chunked import DocumentChunkedAttention

    saved = []
    for block in gm.model.blocks.values():
        attn = block.attention
        if isinstance(attn, DocumentChunkedAttention):
            saved.append((attn, attn.cross_doc_mode, attn._pattern))
            attn.cross_doc_mode = "standard"
            attn._pattern = AttentionPattern(name="standard")
    try:
        yield
    finally:
        for attn, mode, pattern in saved:
            attn.cross_doc_mode = mode
            attn._pattern = pattern


@torch.no_grad()
def generate_batch_docchunk(
    gm,
    prefills: List[List[int]],
    *,
    device: torch.device,
    eos_id: int,
    pad_token_id: int,
    max_new_tokens: int,
    max_length: int,
    is_answer_complete: Callable[[List[int]], bool],
) -> List[List[int]]:
    """
    Batched greedy decode, faithful to the bs=1 loop in ``eval_lc_native_docchunk*.py``:

    .. code-block:: python

        if nxt == eos_id: break
        new_content.append(nxt)
        if <per-task stop condition on new_content>: break
        nxt = argmax(model([[nxt]]))

    generalized per-row: once a row hits its stop condition, it stops appending to its own
    ``new_content`` but keeps getting fed a dummy ``eos_id`` token every subsequent step (to keep the
    batched forward call rectangular) -- inert, since that row's output is never read again. Greedy
    argmax decoding is deterministic, so a correct implementation matches bs=1 exactly (see
    ``debug/batched_eval_parity/`` for the proof).

    :param gm: The built ``TransformerGenerationModule`` (already configured: ``variant in
        ("dense", "full")`` chunk-attention state must already be set up by the caller --
        ``enable_document_chunk_attention`` for "dense", plus :func:`force_standard_pattern` wrapping
        this call for "full").
    :param prefills: One prompt token-id list per example in this batch (already length-filtered by
        the caller's ``cap`` check).
    :param eos_id: The document-separator / EOS id that ends generation.
    :param pad_token_id: The reserved id used for the left-pad prefix (must match what the caller
        passed as ``enable_document_chunk_attention(pad_id=...)`` so padding is excluded from the
        chunked mask too).
    :param max_new_tokens: Per-task decode budget.
    :param max_length: KV-cache capacity (same semantics as the bs=1 path's ``--max-length``).
    :param is_answer_complete: Called with the row's ``new_content`` (token ids so far, most-recent
        last) immediately after a token is appended; returning ``True`` stops that row (mirrors each
        eval file's own ``should_stop`` / ``_answer_complete``).

    :returns: One token-id list per example (the generated content only, no prompt, no EOS/pad) --
        callers ``tok.decode`` + post-process (``</think>`` split etc.) exactly as the bs=1 path does.
    """
    B = len(prefills)
    padded, leftpad = left_pad_batch(prefills, pad_token_id, device)
    gm.prepare_inference_cache(B, max_length)

    logits = gm.model(padded, logits_to_keep=1, cache_leftpad=leftpad)
    cur = logits[:, -1].argmax(dim=-1)  # (B,) -- every row's real content ends at the last column

    finished = torch.zeros(B, dtype=torch.bool, device=device)
    new_content: List[List[int]] = [[] for _ in range(B)]
    eos_tensor = torch.full((B,), eos_id, dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        is_eos = cur.eq(eos_id)
        append_now = (~finished) & (~is_eos)
        finished = finished | is_eos
        cur_list = cur.tolist()
        for i in range(B):
            if append_now[i]:
                new_content[i].append(cur_list[i])
                if is_answer_complete(new_content[i]):
                    finished[i] = True
        if bool(finished.all()):
            break
        feed = torch.where(finished, eos_tensor, cur)
        logits = gm.model(feed.view(B, 1), logits_to_keep=1)
        cur = logits[:, -1].argmax(dim=-1)

    return new_content
