"""
Chunked-document attention masks -- a port of the corpus-reasoning ``chunked_attention`` pattern
family into OLMo-core.

Each token carries a ``chunk_id`` role:

  * ``>= 0`` -- the index of the document/chunk the token belongs to (context chunks);
  * ``-1`` (``FREE_CHUNK_ID``) -- a "free" token (query / answer / instruction) that attends to
    everything and is attended to by everything;
  * ``-2`` (``PAD_CHUNK_ID``) -- padding (never attends or is attended);
  * ``-3`` (``SINK_CHUNK_ID``) -- the instruction/prompt prefix before the first document (a global
    sink; treated like FREE here, kept distinct for future policies).

The universal rule for every pattern is::

    allowed = causal & not_pad & (context_ok | q_free | kv_free)

so FREE tokens always bridge across chunks; only *context-context* edges are restricted, by the
selected :class:`AttentionPattern`. This mirrors corpus-reasoning
``scripts/lib/chunked_attention.py`` (``AttentionPattern`` / ``build_dense_bool_mask``); the OLMo-core
landmark variant folds the resulting boolean mask into its grouped-softmax additive mask (see
:class:`~olmo_core.nn.attention.landmark_document.DocumentLandmarkAttention`).
"""

from dataclasses import dataclass
from typing import Optional

import torch

__all__ = [
    "PAD_CHUNK_ID",
    "FREE_CHUNK_ID",
    "SINK_CHUNK_ID",
    "CHUNKED_ATTENTION_PATTERNS",
    "AttentionPattern",
    "hierarchical_effective_layer",
    "build_chunked_allowed_mask",
    "build_chunked_mask_mod",
    "build_chunk_ids_from_tokens",
    "build_is_anchor",
]

# Chunk-id role conventions (shared by data prep, training, and eval).
PAD_CHUNK_ID = -2
FREE_CHUNK_ID = -1
SINK_CHUNK_ID = -3

CHUNKED_ATTENTION_PATTERNS = (
    "standard",
    "chunked",
    "doc_window",
    "last_token_anchor",
    "token_window",
    "random_token",
    "hierarchical_dilated",
)


@dataclass
class AttentionPattern:
    """
    Configuration for one chunked-attention pattern variant. All patterns preserve the invariant that
    FREE tokens (``chunk_id == -1``: query / answer / instruction) attend to everything and are
    attended to by everything; the parameters below only affect *context-context* edges.

    :param name: One of :data:`CHUNKED_ATTENTION_PATTERNS`.
    :param doc_window_k: ``"doc_window"``: document ``i`` attends to documents ``[i - k, i]``.
    :param token_window_w: ``"token_window"``: raw token-level causal window width (may cross chunk
        boundaries).
    :param keep_prob: ``"random_token"``: Bernoulli keep-probability for each cross-chunk
        ``(q_tok, k_tok)`` edge. ``0.0`` collapses to ``"chunked"``; ``1.0`` to ``"standard"``.
    :param random_seed: Seed for the ``"random_token"`` Bernoulli sample (combine with an example
        index upstream for per-example determinism).
    :param dilation_n: ``"hierarchical_dilated"``: number of documents a context query attends per
        layer (itself + the ``n-1`` strided predecessors). ``n == 1`` collapses to ``"chunked"``.
    :param dilation_m: ``"hierarchical_dilated"``: dilation base ``m >= 1``. At transformer layer
        ``ell`` the stride is ``s = m**ell`` (saturated, see :func:`build_chunked_allowed_mask`), so
        the receptive span is ``(n-1)*m**ell`` documents. ``m == 1`` collapses to a fixed
        ``"doc_window"`` of width ``n-1`` at every layer.
    :param dilation_max_docs: ``"hierarchical_dilated"``: optional fixed reference document count for
        the saturation cap. When ``None`` (default) the cap is computed **per sequence** from the
        actual maximum context-chunk index; when set, this fixed value is used instead (so every
        sequence saturates at the same layer).
    """

    name: str = "chunked"
    doc_window_k: int = 0
    token_window_w: int = 0
    keep_prob: float = 1.0
    random_seed: int = 42
    dilation_n: int = 2
    dilation_m: int = 2
    dilation_max_docs: Optional[int] = None

    def __post_init__(self) -> None:
        if self.name not in CHUNKED_ATTENTION_PATTERNS:
            raise ValueError(
                f"Unknown chunked attention pattern {self.name!r}; expected one of "
                f"{CHUNKED_ATTENTION_PATTERNS}"
            )
        if self.name == "hierarchical_dilated":
            if self.dilation_n < 1:
                raise ValueError(
                    f"hierarchical_dilated requires dilation_n >= 1 (got {self.dilation_n})"
                )
            if self.dilation_m < 1:
                raise ValueError(
                    f"hierarchical_dilated requires dilation_m >= 1 (got {self.dilation_m})"
                )

    def needs_anchor(self) -> bool:
        return self.name == "last_token_anchor"


def build_is_anchor(input_ids: torch.Tensor, doc_end_id: int) -> torch.Tensor:
    """
    Mark ``<|doc_end|>`` positions as per-document anchors (the ``"last_token_anchor"`` pattern).

    By the time the doc-end boundary token is emitted, causal attention has folded every token of its
    document into its hidden state, so it serves as a per-document summary that later tokens can
    attend to.

    :param input_ids: Token ids, shape ``(..., S)``.
    :param doc_end_id: The ``<|doc_end|>`` token id.

    :returns: A boolean tensor the same shape as ``input_ids``.
    """
    return input_ids == doc_end_id


def build_chunk_ids_from_tokens(
    input_ids: torch.Tensor,
    doc_start_id: int,
    doc_end_id: int,
    eos_id: int,
    mode: str = "chunked",
    pad_id: Optional[int] = None,
) -> torch.Tensor:
    """
    Reconstruct per-token ``chunk_id`` roles ``(B, S)`` from the token stream at runtime.

    Vectorized port of corpus-reasoning ``build_roles``. Each ``<|doc_start|> ... <|doc_end|>`` span
    (markers included) is one context chunk (monotonic doc index); tokens outside any span are
    ``FREE``; everything strictly after the first ``eos_id`` in a row is ``PAD`` (one EOS-terminated
    example per padded instance); and, when ``mode == "modified_swa"``, the FREE prefix before the
    first ``<|doc_start|>`` is marked ``SINK``.

    :param input_ids: Token ids, shape ``(B, S)`` (or ``(S,)``).
    :param doc_start_id: The ``<|doc_start|>`` token id.
    :param doc_end_id: The ``<|doc_end|>`` token id.
    :param eos_id: The EOS / document-terminator token id (everything after the first one is pad).
    :param mode: ``"chunked"`` (no SINK) or ``"modified_swa"`` (mark the prefix SINK).
    :param pad_id: Optional dedicated padding token id (e.g. the interior window-fill padding inserted
        by :func:`~olmo_core.data.document_chunk_landmark.emit_document_chunk_landmark`). When given,
        every position holding this id is marked ``PAD`` -- so window-fill padding is non-attendable
        rather than treated as a FREE token. Must differ from every content/marker id.

    :returns: An int32 tensor of shape ``(B, S)`` with the role of each token.
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    B, S = input_ids.shape
    device = input_ids.device
    pos = torch.arange(S, device=device).expand(B, S)

    starts = input_ids == doc_start_id
    ends = input_ids == doc_end_id
    # Number of opened starts vs closed ends up to (and including) each position. A token is "inside"
    # a document (markers included) iff more spans have been opened than closed *before* this token --
    # which keeps the closing ``<|doc_end|>`` itself attached to its document.
    n_start = torch.cumsum(starts.to(torch.long), dim=1)
    n_end = torch.cumsum(ends.to(torch.long), dim=1)
    inside = n_start > (n_end - ends.to(torch.long))
    chunk_ids = torch.where(inside, n_start - 1, torch.full_like(n_start, FREE_CHUNK_ID))

    # PAD = everything strictly after the first EOS in the row.
    is_eos = input_ids == eos_id
    eos_pos = torch.where(is_eos, pos, torch.full_like(pos, S))
    pad_from = eos_pos.min(dim=1, keepdim=True).values + 1  # (B, 1)
    chunk_ids = torch.where(pos >= pad_from, torch.full_like(chunk_ids, PAD_CHUNK_ID), chunk_ids)

    # Dedicated interior padding (window fill) -> PAD, so it is never attended/attending.
    if pad_id is not None:
        chunk_ids = torch.where(
            input_ids == pad_id, torch.full_like(chunk_ids, PAD_CHUNK_ID), chunk_ids
        )

    if mode == "modified_swa":
        has_start = starts.any(dim=1, keepdim=True)
        start_pos = torch.where(starts, pos, torch.full_like(pos, S))
        first_start = start_pos.min(dim=1, keepdim=True).values  # (B, 1); S if no start
        # SINK only applies to rows that actually contain a document (matches build_roles).
        sink = has_start & (pos < first_start) & (chunk_ids == FREE_CHUNK_ID)
        chunk_ids = torch.where(sink, torch.full_like(chunk_ids, SINK_CHUNK_ID), chunk_ids)

    return chunk_ids.to(torch.int32)


def hierarchical_effective_layer(
    layer_idx: int, n: int, m: int, max_chunk: torch.Tensor
) -> torch.Tensor:
    """
    Per-sequence *saturated* (effective) layer index for the ``"hierarchical_dilated"`` pattern.

    The dilation stride at transformer layer ``ell`` is ``m**ell`` and the receptive span of a layer
    is ``(n-1)*m**ell`` documents. Once that span already covers all of a sequence's history there is
    nothing left to dilate into, so deeper layers reuse the widest pattern instead of expanding past
    the end of the document list. Concretely, ``L* = min{ ell : (n-1)*m**ell >= max_chunk }`` and the
    effective layer is ``min(layer_idx, L*)``.

    :param layer_idx: The transformer layer index (0-based).
    :param n: Documents attended per layer (``dilation_n``).
    :param m: Dilation base (``dilation_m``).
    :param max_chunk: Per-sequence maximum context-chunk index, shape ``(B,)``.

    :returns: An integer tensor ``(B,)`` of effective layer indices in ``[0, layer_idx]``.
    """
    cap = torch.full_like(max_chunk, layer_idx)
    # For ``m == 1`` the stride is constant (``1**ell == 1``) and for ``n == 1`` only the own document
    # is ever in range, so saturation is a no-op -- the stride ``m**layer_idx`` already behaves
    # correctly and we keep ``cap = layer_idx``.
    if n > 1 and m > 1:
        found = torch.zeros_like(max_chunk, dtype=torch.bool)
        span = n - 1
        for ell in range(layer_idx + 1):
            cover = max_chunk <= span
            newly = cover & ~found
            cap = torch.where(newly, torch.full_like(cap, ell), cap)
            found = found | cover
            if bool(found.all()):
                break
            span *= m
    return cap


def build_chunked_allowed_mask(
    pattern: AttentionPattern,
    chunk_ids: torch.Tensor,
    is_anchor: Optional[torch.Tensor] = None,
    random_keep: Optional[torch.Tensor] = None,
    layer_idx: int = 0,
) -> torch.Tensor:
    """
    Materialize a chunked-attention pattern as a dense boolean ``(B, S, S)`` mask (``True`` = attend).

    Port of corpus-reasoning ``build_dense_bool_mask``. ``allowed = causal & not_pad & (context_ok |
    q_free | kv_free)``, where ``context_ok`` depends on the pattern.

    :param pattern: The :class:`AttentionPattern`.
    :param chunk_ids: Per-token role ids, shape ``(B, S)`` or ``(S,)``. See module docstring.
    :param is_anchor: ``(B, S)`` / ``(S,)`` bool, required for ``"last_token_anchor"``.
    :param random_keep: ``(B, S, S)`` / ``(S, S)`` bool, required for ``"random_token"``.
    :param layer_idx: The transformer layer index; only used by the layer-dependent
        ``"hierarchical_dilated"`` pattern (the stride is ``m**layer_idx``, saturated). Ignored by all
        other patterns.

    :returns: A boolean ``(B, S, S)`` tensor; ``True`` where the query (dim 1) may attend the key
        (dim 2).
    """
    if chunk_ids.dim() == 1:
        chunk_ids = chunk_ids.unsqueeze(0)
    B, S = chunk_ids.shape
    device = chunk_ids.device

    q = torch.arange(S, device=device)
    kv = torch.arange(S, device=device)
    causal = q.unsqueeze(1) >= kv.unsqueeze(0)  # (S, S)

    qc = chunk_ids.unsqueeze(2)  # (B, S, 1)
    kc = chunk_ids.unsqueeze(1)  # (B, 1, S)
    q_not_pad = qc != PAD_CHUNK_ID
    kv_not_pad = kc != PAD_CHUNK_ID
    q_free = qc < 0  # FREE or SINK: globally attending
    kv_free = kc < 0  # FREE or SINK: globally attendable
    # ``< 0`` lumps PAD in too, but the ``not_pad`` gates below remove pad rows/cols, so a PAD key is
    # never attended and a PAD query never attends regardless of the free shortcut.
    q_free = q_free & q_not_pad
    kv_free = kv_free & kv_not_pad
    same_chunk = (qc == kc) & (qc >= 0)

    name = pattern.name
    if name == "standard":
        return causal.unsqueeze(0) & q_not_pad & kv_not_pad

    if name == "chunked":
        context_ok = same_chunk
    elif name == "doc_window":
        diff = qc - kc
        context_ok = (diff >= 0) & (diff <= pattern.doc_window_k) & (qc >= 0) & (kc >= 0)
    elif name == "last_token_anchor":
        if is_anchor is None:
            raise ValueError("last_token_anchor requires an is_anchor tensor")
        if is_anchor.dim() == 1:
            is_anchor = is_anchor.unsqueeze(0)
        anchor_kv = is_anchor.unsqueeze(1) & (kc >= 0)  # (B, 1, S)
        context_ok = same_chunk | anchor_kv
    elif name == "token_window":
        tok_diff = q.unsqueeze(1) - kv.unsqueeze(0)  # (S, S)
        tok_ok = ((tok_diff >= 0) & (tok_diff <= pattern.token_window_w)).unsqueeze(0)
        context_ok = same_chunk | (tok_ok & (qc >= 0) & (kc >= 0))
    elif name == "random_token":
        if random_keep is None:
            raise ValueError("random_token requires a random_keep tensor")
        if random_keep.dim() == 2:
            random_keep = random_keep.unsqueeze(0)
        cross_doc = (qc != kc) & (qc >= 0) & (kc >= 0)
        context_ok = same_chunk | (cross_doc & random_keep)
    elif name == "hierarchical_dilated":
        n = pattern.dilation_n
        m = pattern.dilation_m
        # Per-sequence max context-chunk index (rows with no context chunks fall back to 0).
        is_ctx = chunk_ids >= 0  # (B, S)
        max_chunk = torch.where(is_ctx, chunk_ids, torch.zeros_like(chunk_ids)).amax(dim=1)  # (B,)
        if pattern.dilation_max_docs is not None:
            max_chunk = torch.full_like(max_chunk, pattern.dilation_max_docs)
        eff_l = hierarchical_effective_layer(layer_idx, n, m, max_chunk)  # (B,)
        # Stride s = m**eff_l per sequence (>= 1). eff_l is capped at layer_idx so this never overflows
        # for any sane depth.
        stride = (torch.full_like(eff_l, m) ** eff_l).clamp(min=1).view(B, 1, 1)  # (B, 1, 1)
        diff = qc - kc  # (B, S, S): chunk-index gap (query chunk - key chunk)
        # Attend the n documents at stride s behind (and including) the query's own document: the gap
        # must be a non-negative multiple of s and within the first n strided steps.
        stride_ok = (
            (diff >= 0) & (qc >= 0) & (kc >= 0) & ((diff % stride) == 0) & ((diff // stride) < n)
        )
        context_ok = same_chunk | stride_ok
    else:  # pragma: no cover - guarded by AttentionPattern.__post_init__
        raise ValueError(f"Unknown chunked attention pattern: {name}")

    allowed = causal.unsqueeze(0) & q_not_pad & kv_not_pad & (context_ok | q_free | kv_free)
    # NaN guard: a fully-masked query row (e.g. a PAD position) would make softmax produce NaN. Always
    # let a query attend itself; such rows are dropped by the loss mask anyway. (Mirrors the diagonal
    # term in corpus-reasoning's modified_swa mask_mod.)
    diag = torch.eye(S, dtype=torch.bool, device=device).unsqueeze(0)
    return allowed | diag


def build_chunked_mask_mod(pattern: AttentionPattern, chunk_ids: torch.Tensor):
    """
    Build a FlexAttention ``mask_mod`` closure for ``pattern`` over per-token ``chunk_ids`` ``(B, S)``,
    or return ``None`` if the pattern needs extra per-edge tensors not expressible as a pure
    ``mask_mod`` (``last_token_anchor`` / ``token_window`` / ``random_token`` -- callers fall back to
    the dense :func:`build_chunked_allowed_mask`).

    The returned ``mask_mod(b, h, q_idx, kv_idx) -> bool`` is element-equivalent to
    :func:`build_chunked_allowed_mask` (same ``causal & not_pad & (context_ok | q_free | kv_free)``
    rule plus the self-diagonal NaN guard), so a block-sparse FlexAttention kernel computes exactly the
    same masked softmax as the dense path -- but skips fully-masked blocks. See
    :class:`~olmo_core.nn.attention.document_chunked.DocumentChunkedAttention`.

    :param pattern: The :class:`AttentionPattern` (``chunked`` / ``standard`` / ``doc_window``).
    :param chunk_ids: Per-token role ids ``(B, S)`` on the target device.
    """
    if chunk_ids.dim() == 1:
        chunk_ids = chunk_ids.unsqueeze(0)
    cids = chunk_ids
    name = pattern.name

    if name == "standard":
        # NB: build_chunked_allowed_mask returns "standard" WITHOUT the diagonal guard (plain causal,
        # padding-aware), so neither does this mask_mod -- they must stay element-identical.
        def mask_mod(b, h, q_idx, kv_idx):
            qc = cids[b, q_idx]
            kc = cids[b, kv_idx]
            return (q_idx >= kv_idx) & (qc != PAD_CHUNK_ID) & (kc != PAD_CHUNK_ID)

        return mask_mod

    if name == "chunked":

        def mask_mod(b, h, q_idx, kv_idx):
            qc = cids[b, q_idx]
            kc = cids[b, kv_idx]
            q_np = qc != PAD_CHUNK_ID
            kv_np = kc != PAD_CHUNK_ID
            same = (qc == kc) & (qc >= 0)
            q_free = (qc < 0) & q_np
            kv_free = (kc < 0) & kv_np
            return ((q_idx >= kv_idx) & q_np & kv_np & (same | q_free | kv_free)) | (
                q_idx == kv_idx
            )

        return mask_mod

    if name == "doc_window":
        k_win = pattern.doc_window_k

        def mask_mod(b, h, q_idx, kv_idx):
            qc = cids[b, q_idx]
            kc = cids[b, kv_idx]
            q_np = qc != PAD_CHUNK_ID
            kv_np = kc != PAD_CHUNK_ID
            diff = qc - kc
            ctx_ok = (diff >= 0) & (diff <= k_win) & (qc >= 0) & (kc >= 0)
            q_free = (qc < 0) & q_np
            kv_free = (kc < 0) & kv_np
            return ((q_idx >= kv_idx) & q_np & kv_np & (ctx_ok | q_free | kv_free)) | (
                q_idx == kv_idx
            )

        return mask_mod

    return None  # unsupported pattern -> caller uses the dense materialized mask
