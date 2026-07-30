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

import random
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
    "chunk_token_offset",
    "build_chunked_allowed_mask",
    "build_chunked_mask_mod",
    "build_chunk_ids_from_tokens",
    "build_is_anchor",
    "mask_mix_standard_prob",
    "collapse_roles_to_causal",
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
    "random_doc",
    "hierarchical_dilated",
    "summary_attention",
    "gold_hop_controlled",
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
    :param random_seed: Seed for the ``"random_token"`` / ``"random_doc"`` pseudo-random sample.
    :param doc_keep_prob: ``"random_doc"``: each context document attends to itself + a random subset
        of the **strictly-earlier** documents, where each earlier document is kept independently with
        this Bernoulli probability (so ~``doc_keep_prob`` of previous docs on average). The keep set is
        a deterministic seeded hash of the ordered ``(query_doc, key_doc)`` pair (fixed random sparsity
        graph over document indices, à la BigBird's random attention; vary ``random_seed`` for a
        different graph), so it is identical across layers and reproducible. ``0.0`` collapses to
        ``"chunked"`` (isolated docs); ``1.0`` to full causal cross-document attention.
    :param dilation_n: ``"hierarchical_dilated"``: number of documents a context query attends per
        layer (itself + the ``n-1`` strided predecessors). ``n == 1`` collapses to ``"chunked"``.
    :param dilation_m: ``"hierarchical_dilated"``: dilation base ``m >= 1``. At cycle position ``p``
        (see :attr:`dilation_cycle`) the stride is ``s = m**p`` (saturated within the cycle, see
        :func:`build_chunked_allowed_mask`), so the receptive span is ``(n-1)*m**p`` documents.
        ``m == 1`` collapses to a fixed ``"doc_window"`` of width ``n-1`` at every layer.
    :param dilation_cycle: ``"hierarchical_dilated"``: the **rotation period** ``L``. The per-layer
        dilation stride *rotates* with a fixed period -- the cycle position is ``p = layer_idx % L`` --
        so deep layers revisit the fine-grained small-stride patterns instead of freezing at the widest
        stride (the "Hierarchical K" schedule; see :func:`hierarchical_effective_layer`). Defaults to
        3. ``dilation_cycle=None`` (the old pure-saturation schedule, no rotation) is **deprecated** and
        raises :class:`NotImplementedError`.
    :param dilation_max_docs: ``"hierarchical_dilated"``: optional fixed reference document count for
        the within-cycle saturation cap. When ``None`` (default) the cap is computed **per sequence**
        from the actual maximum context-chunk index; when set, this fixed value is used instead (so
        every sequence saturates at the same cycle position).
    :param summary_every_k: ``"summary_attention"``: the **cell size** -- the number of context
        documents per cell. The tokenized layout must place exactly ``summary_every_k`` documents then
        one **summary span** in each cell, so chunk indices run on a stride of ``summary_every_k + 1``
        (``cell(id) = id // (k+1)``, ``is_summary(id) = (id % (k+1)) == k``). Must be ``>= 1``, and the
        document count must be divisible by it so no partial cell breaks the modular arithmetic.
    :param summary_bandwidth: ``"summary_attention"``: the **relay bandwidth** ``b`` -- how many of each
        earlier summary span's leading tokens a later chunk may attend. This is the dose knob of the
        bandwidth ladder: the tokenized data is unchanged across rungs, only visibility varies.
        ``0`` removes the relay entirely, which reproduces a pure "cell blocks" mask (documents see
        their own cell only) and is the ladder's floor control; large values expose the whole span.
    :param summary_relay: ``"summary_attention"``: whether a summary span may read its own cell's
        documents. ``True`` (default) is the treatment -- the span aggregates its cell and relays it
        forward. ``False`` is the **placebo**: the span keeps its position, its tokens, and every edge
        into it, but reads nothing, so it provably carries **zero** document content (see
        :func:`build_chunked_allowed_mask`). Used to test whether information-free keys alone move the
        metric.
    :param gold_hops: ``"gold_hop_controlled"``: which arm of the multi-hop gold-routing ladder --
        ``1`` (the gold edge is forced **present**), ``2`` / ``3`` (the gold edge is **deleted** and the
        shortest gold path forced to exactly that length), or
        :data:`~olmo_core.nn.attention.gold_hop_mask.GOLD_HOPS_INF` (``-1``: gold edge deleted **and**
        every path cut -- the leak-matched control). This pattern is a pure **consumer**: the graph is
        built per example by :mod:`~olmo_core.nn.attention.gold_hop_mask` (which owns the base
        ``doc_keep_prob``, the seed, and the gold-pair sidecar) and handed to
        :func:`build_chunked_allowed_mask` as ``doc_adjacency``. The field is carried here so the arm
        is recorded in the saved ``config.json`` and can be cross-checked against the installed hook
        instead of trusted.
    :param gold_decoys: ``"gold_hop_controlled"``: distance-matched NON-gold pairs per gold pair given
        the identical edit, so the arm's structural signature stops naming the gold pair. Carried here
        for provenance; the graph is built by
        :mod:`~olmo_core.nn.attention.gold_hop_mask`. ``0`` is the un-camouflaged design, whose leak is
        measured and large (``hop_inf``: a graph-only classifier reaches precision@3 16.2% vs 0.245%
        chance). ``12`` cuts that to 2.0% and makes ``hop2`` / ``hop_inf`` leak-matched.
    """

    name: str = "chunked"
    doc_window_k: int = 0
    token_window_w: int = 0
    keep_prob: float = 1.0
    doc_keep_prob: float = 0.1
    random_seed: int = 42
    random_doc_per_example: bool = False
    dilation_n: int = 2
    dilation_m: int = 2
    dilation_cycle: Optional[int] = 3
    dilation_max_docs: Optional[int] = None
    summary_every_k: int = 10
    summary_bandwidth: int = 0
    summary_relay: bool = True
    gold_hops: int = 2
    gold_decoys: int = 0

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
            if self.dilation_cycle is None:
                raise NotImplementedError(
                    "hierarchical_dilated 'dilation_cycle=None' (the pure-saturation schedule with no "
                    "rotation) is deprecated. Set a fixed rotation period, e.g. dilation_cycle=3 "
                    "(the default), so the dilation stride rotates with depth."
                )
            if self.dilation_cycle < 1:
                raise ValueError(
                    f"hierarchical_dilated requires dilation_cycle >= 1 (got {self.dilation_cycle})"
                )
        if self.name == "random_doc" and not (0.0 <= self.doc_keep_prob <= 1.0):
            raise ValueError(
                f"random_doc requires 0 <= doc_keep_prob <= 1 (got {self.doc_keep_prob})"
            )
        if self.name == "summary_attention":
            if self.summary_every_k < 1:
                raise ValueError(
                    f"summary_attention requires summary_every_k >= 1 (got {self.summary_every_k})"
                )
            if self.summary_bandwidth < 0:
                raise ValueError(
                    f"summary_attention requires summary_bandwidth >= 0 "
                    f"(got {self.summary_bandwidth})"
                )
        if self.name == "gold_hop_controlled" and self.gold_hops not in (1, 2, 3, -1):
            # -1 == gold_hop_mask.GOLD_HOPS_INF; spelled out to keep this module import-free of the
            # gold-aware layer (gold_hop_mask imports THIS module).
            raise ValueError(
                f"gold_hop_controlled requires gold_hops in (1, 2, 3, -1 [=inf]) "
                f"(got {self.gold_hops})"
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


# ---------------------------------------------------------------------------
# Mask mixing (runtime schedule)
# ---------------------------------------------------------------------------
#
# "Mask mixing" collapses a randomly-chosen subset of examples in each forward from the chunked
# (block-sparse) mask to **plain causal** (full) attention. The collapse is done purely on the
# per-token roles: setting an example's ``chunk_ids`` to all-FREE (keeping PAD) makes every
# ``mask_mod`` / ``build_chunked_allowed_mask`` pattern degenerate to ``causal & not_pad`` -- i.e.
# plain causal -- with NO separate full-mask code path (the same mask machinery neutralizes itself).
# This mirrors corpus-reasoning ``scripts/lib/olmo_flex_attention.py`` (the ``roles -> FREE`` trick).
#
# The per-example collapse probability ``p`` is either **static** (``standard_mix_prob``, constant
# every forward) or a **linear curriculum** (``mix_start_p -> mix_end_p`` over ``mix_total_forwards``
# microbatch-forwards) -- start "easy" (mostly full attention) and harden into the sparse mask. It is a
# RUNTIME schedule driven by a forward counter (not baked at tokenize time). ``p == 0`` (the default)
# leaves ``chunk_ids`` untouched, so pure-chunked training is bit-identical.


def mask_mix_standard_prob(
    forward_idx: int,
    *,
    standard_mix_prob: float = 0.0,
    mix_start_p: float = 0.0,
    mix_end_p: float = 0.0,
    mix_total_forwards: int = 0,
) -> float:
    """
    Per-example probability of collapsing an example to plain causal on this forward.

    :param forward_idx: The 0-based microbatch-forward index (drives the curriculum anneal).
    :param standard_mix_prob: Static mix probability -- constant on every forward. When ``> 0`` it takes
        precedence and the curriculum params are ignored.
    :param mix_start_p: Curriculum start probability (at ``forward_idx == 0``).
    :param mix_end_p: Curriculum end probability (at ``forward_idx >= mix_total_forwards``).
    :param mix_total_forwards: Number of forwards over which the curriculum anneals linearly. ``0``
        disables the curriculum.

    :returns: The collapse probability ``p`` for this forward (``0.0`` when no mixing is configured).
    """
    if standard_mix_prob > 0.0:
        return standard_mix_prob
    if mix_total_forwards > 0:
        prog = min(1.0, forward_idx / mix_total_forwards)
        return mix_start_p + (mix_end_p - mix_start_p) * prog
    return 0.0


def collapse_roles_to_causal(
    chunk_ids: torch.Tensor,
    p: float,
    *,
    forward_idx: int,
    mix_seed: int = 0,
) -> torch.Tensor:
    """
    With a seeded per-example probability ``p``, collapse an example's roles to all-FREE (keeping PAD),
    which makes any chunked ``mask_mod`` / allowed-mask degenerate to **plain causal** for that example.

    The per-``(forward_idx, example)`` coin is seeded by ``mix_seed`` (a string seed, matching
    corpus-reasoning) so a resumed / rerun job makes identical choices. ``p <= 0`` returns the input
    tensor unchanged (and no clone) so pure-chunked training stays bit-identical.

    :param chunk_ids: Per-token role ids ``(B, S)`` (or ``(S,)``). See the module docstring.
    :param p: The collapse probability from :func:`mask_mix_standard_prob`.
    :param forward_idx: The forward index used in the per-example seed.
    :param mix_seed: Base seed for the deterministic per-example coin.

    :returns: ``chunk_ids`` (possibly a new, partially-collapsed clone; the input is never mutated).
    """
    if p <= 0.0:
        return chunk_ids
    if chunk_ids.dim() == 1:
        chunk_ids = chunk_ids.unsqueeze(0)
    B = chunk_ids.shape[0]
    out: Optional[torch.Tensor] = None
    for b in range(B):
        # str seed (tuples are not a supported Random seed type); deterministic across runs/platforms.
        if random.Random(f"{mix_seed}:{forward_idx}:{b}").random() < p:
            if out is None:
                out = chunk_ids.clone()
            rb = out[b]
            out[b] = torch.where(rb == PAD_CHUNK_ID, rb, torch.full_like(rb, FREE_CHUNK_ID))
    return chunk_ids if out is None else out


def hierarchical_effective_layer(
    layer_idx: int, n: int, m: int, max_chunk: torch.Tensor, cycle: Optional[int] = None
) -> torch.Tensor:
    """
    Per-sequence *effective* layer index (cycle position) for the ``"hierarchical_dilated"`` pattern.

    The dilation stride at transformer layer ``ell`` is ``m**ell`` and the receptive span of a layer
    is ``(n-1)*m**ell`` documents.

    Two effects combine, in order:

    * **Rotation** (``cycle`` given): the schedule *rotates* with a fixed period so deep layers revisit
      the fine-grained (small-stride) patterns instead of freezing at the widest stride. The cycle
      position is ``p = layer_idx % cycle`` (à la the "Hierarchical K" positional schedule). With
      ``cycle=None`` the position is just ``layer_idx`` (no rotation).
    * **Within-cycle saturation**: once a cycle position's span already covers all of a sequence's
      history there is nothing left to dilate into, so it is capped at
      ``L* = min{ ell : (n-1)*m**ell >= max_chunk }``. Big sequences (large ``L*``) never hit the cap
      and rotate freely; small ones saturate at the right position.

    So the returned effective layer is ``min(layer_idx % cycle, L*)`` (or ``min(layer_idx, L*)`` when
    ``cycle`` is ``None``).

    :param layer_idx: The transformer layer index (0-based).
    :param n: Documents attended per layer (``dilation_n``).
    :param m: Dilation base (``dilation_m``).
    :param max_chunk: Per-sequence maximum context-chunk index, shape ``(B,)``.
    :param cycle: The rotation period (``dilation_cycle``). ``None`` disables rotation (pure
        saturation) -- retained only for internal/low-level callers; the user-facing
        :class:`AttentionPattern` deprecates ``dilation_cycle=None``.

    :returns: An integer tensor ``(B,)`` of effective layer indices in ``[0, min(layer_idx, cycle-1)]``.
    """
    layer_idx = int(layer_idx)
    if cycle is not None:
        layer_idx = layer_idx % int(cycle)
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


# Multiplicative-hash constants for the ``"random_doc"`` per-document-pair keep (spatial-hash style).
# Kept module-level so the dense mask and the FlexAttention ``mask_mod`` compute the SAME pseudo-random
# value for a given ``(query_doc, key_doc, seed)`` -> identical masks on both paths.
_RD_A = 73856093
_RD_B = 19349663
_RD_C = 83492791
_RD_MIX = 2654435761
_RD_MASK = 0x7FFFFFFF  # 2**31 - 1


def random_doc_nonce(chunk_ids: torch.Tensor) -> torch.Tensor:
    """Per-example nonce for the ``"random_doc"`` pattern's *per-example* mode.

    Derived from the example's own ``chunk_ids`` content (the document-boundary layout, which differs
    between examples because documents differ in length), so it is:

    * **distinct per example** -- every example gets its own random sparsity graph, instead of the one
      global graph over document *indices* that the default mode shares across every example; and
    * **deterministic** -- the same example always yields the same graph, at train and at eval, so a
      held-out example is scored on a well-defined mask rather than a fresh coin flip.

    This is the knob for the ablation "does the model need a *stable* sparsity graph to learn, or is
    sparse-but-varied connectivity enough?" -- the default (shared graph) lets the model potentially
    learn the fixed structure; the per-example mode denies it that.

    :param chunk_ids: ``(B, T)`` per-token chunk roles.

    :returns: An int64 tensor of shape ``(B,)``.
    """
    b, t = chunk_ids.shape
    pos = torch.arange(t, device=chunk_ids.device, dtype=torch.int64) + 1
    # +3 lifts the negative roles (FREE=-1, PAD=-2, SINK=-3) to >=0 so they contribute to the hash.
    h = ((chunk_ids.to(torch.int64) + 3) * pos * _RD_A) ^ (pos * _RD_B)
    return (h.sum(dim=-1) * _RD_MIX) & _RD_MASK


def chunk_token_offset(chunk_ids: torch.Tensor) -> torch.Tensor:
    """
    Per-token offset within its own chunk, i.e. ``0`` for the first token of each chunk, ``1`` for the
    second, and so on. Required by the ``"summary_attention"`` pattern's **bandwidth gate**, which
    exposes only the leading ``summary_bandwidth`` tokens of each summary span.

    Chunks are contiguous runs of equal ``chunk_id`` (the document-boundary markers guarantee this), so
    the offset is ``position - (position of the first token of this run)``, computed with a running
    maximum over run-start positions. Offsets for FREE / PAD runs are meaningless but harmless: the
    pattern only reads this for context chunks.

    Computed once per forward and closed over by
    :func:`build_chunked_mask_mod`, mirroring :func:`random_doc_nonce`, so the ``mask_mod`` body stays a
    pure elementwise function of ``(b, q_idx, kv_idx)``.

    :param chunk_ids: Per-token role ids ``(B, T)``. See the module docstring.

    :returns: An int64 tensor of shape ``(B, T)``.
    """
    b, t = chunk_ids.shape
    pos = torch.arange(t, device=chunk_ids.device, dtype=torch.int64).expand(b, t)
    is_start = torch.ones_like(chunk_ids, dtype=torch.bool)
    is_start[:, 1:] = chunk_ids[:, 1:] != chunk_ids[:, :-1]
    run_start = torch.cummax(torch.where(is_start, pos, torch.zeros_like(pos)), dim=1).values
    return pos - run_start


def _random_doc_keep(
    qc: torch.Tensor,
    kc: torch.Tensor,
    keep_prob: float,
    seed: int,
    nonce: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Deterministic Bernoulli keep for the ``"random_doc"`` pattern: ``True`` iff the ordered
    document pair ``(query_doc=qc, key_doc=kc)`` is kept, with per-pair probability ``keep_prob``.

    The keep is a seeded multiplicative hash of ``(qc, kc, seed)`` (no layer term), so it is a fixed
    random sparsity graph over document indices -- identical across layers and reproducible. ``qc`` /
    ``kc`` broadcast (e.g. ``(B, S, 1)`` and ``(B, 1, S)`` -> ``(B, S, S)``). Element-equivalent to the
    ``random_doc`` branch of :func:`build_chunked_mask_mod`.

    :param nonce: Optional per-example nonce, shape ``(B,)`` (see :func:`random_doc_nonce`). When
        given it is mixed into the hash, so **each example gets its own sparsity graph** instead of
        one graph shared by every example. Still deterministic per example -- eval reproduces the
        same graph for the same input.
    """
    a = qc.to(torch.int64)
    b = kc.to(torch.int64)
    h = (a * _RD_A) ^ (b * _RD_B) ^ (int(seed) * _RD_C)
    if nonce is not None:
        # broadcast (B,) against the trailing query/key dims of qc/kc
        h = h ^ nonce.to(torch.int64).reshape(-1, *([1] * (h.dim() - 1)))
    h = (h * _RD_MIX) & _RD_MASK
    u = h.to(torch.float32) * (1.0 / float(_RD_MASK))
    return u < keep_prob


def build_chunked_allowed_mask(
    pattern: AttentionPattern,
    chunk_ids: torch.Tensor,
    is_anchor: Optional[torch.Tensor] = None,
    random_keep: Optional[torch.Tensor] = None,
    layer_idx: int = 0,
    doc_adjacency: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Materialize a chunked-attention pattern as a dense boolean ``(B, S, S)`` mask (``True`` = attend).

    Port of corpus-reasoning ``build_dense_bool_mask``. ``allowed = causal & not_pad & (context_ok |
    q_free | kv_free)``, where ``context_ok`` depends on the pattern.

    :param pattern: The :class:`AttentionPattern`.
    :param chunk_ids: Per-token role ids, shape ``(B, S)`` or ``(S,)``. See module docstring.
    :param is_anchor: ``(B, S)`` / ``(S,)`` bool, required for ``"last_token_anchor"``.
    :param random_keep: ``(B, S, S)`` / ``(S, S)`` bool, required for ``"random_token"``.
    :param doc_adjacency: ``(B, D, D)`` / ``(D, D)`` bool, required for ``"gold_hop_controlled"``:
        ``doc_adjacency[b, q, k]`` = example ``b``'s document ``q`` may attend document ``k``. Built
        per example by :mod:`~olmo_core.nn.attention.gold_hop_mask` from a fingerprint lookup, so gold
        identity stays out of the token stream. Ignored by every other pattern.
    :param layer_idx: The transformer layer index; only used by the layer-dependent
        ``"hierarchical_dilated"`` pattern (the stride is ``m**(layer_idx % dilation_cycle)``, saturated
        within the cycle -- see :func:`hierarchical_effective_layer`). Ignored by all
        other patterns.

    :returns: A boolean ``(B, S, S)`` tensor; ``True`` where the query (dim 1) may attend the key
        (dim 2).
    """
    # NOTE (``summary_attention``): the layout must be `k` docs then one summary span per cell, so
    # chunk ids run on a stride of ``k+1``. The pattern is a pure function of ``(qc, kc)`` and the
    # per-token in-chunk offset, with **no layer term** -- so one fixed graph serves every layer and a
    # "hop" is unambiguous, exactly as for ``random_doc``.
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
    elif name == "random_doc":
        # Own document + a seeded-random ~doc_keep_prob subset of STRICTLY EARLIER documents. The keep
        # is a deterministic hash of the ordered (query_doc, key_doc) pair -> a fixed random sparsity
        # graph over document indices, identical across layers (chunk_ids is shared per forward).
        # With random_doc_per_example, a per-example nonce is mixed in so EVERY EXAMPLE gets its own
        # graph (still deterministic per example) -- the "does a stable mask matter?" ablation.
        cross_doc = (qc > kc) & (qc >= 0) & (kc >= 0)
        nonce = random_doc_nonce(chunk_ids) if pattern.random_doc_per_example else None
        keep = _random_doc_keep(qc, kc, pattern.doc_keep_prob, pattern.random_seed, nonce=nonce)
        context_ok = same_chunk | (cross_doc & keep)
    elif name == "hierarchical_dilated":
        n = pattern.dilation_n
        m = pattern.dilation_m
        # Per-sequence max context-chunk index (rows with no context chunks fall back to 0).
        is_ctx = chunk_ids >= 0  # (B, S)
        max_chunk = torch.where(is_ctx, chunk_ids, torch.zeros_like(chunk_ids)).amax(dim=1)  # (B,)
        if pattern.dilation_max_docs is not None:
            max_chunk = torch.full_like(max_chunk, pattern.dilation_max_docs)
        eff_l = hierarchical_effective_layer(
            layer_idx, n, m, max_chunk, cycle=pattern.dilation_cycle
        )  # (B,)
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
    elif name == "summary_attention":
        # Documents are grouped into CELLS of `k` docs, each followed by one SUMMARY span (its own
        # chunk), so chunk indices run on a stride of P = k+1. Within a cell attention is full; the
        # summary span reads its whole cell (it sits at the cell END, so causally it has already seen
        # every doc of the cell) and is then attendable by every LATER cell -> any two documents in
        # different cells are exactly 2 hops apart, with no gold-aware term anywhere.
        p = pattern.summary_every_k + 1
        both_ctx = (qc >= 0) & (kc >= 0)
        cell_q, cell_k = qc // p, kc // p
        is_sum_q = (qc % p) == pattern.summary_every_k
        is_sum_k = (kc % p) == pattern.summary_every_k
        same_cell = (cell_q == cell_k) & both_ctx
        doc_reads_own_cell = same_cell & ~is_sum_q & ~is_sum_k
        # The placebo (`summary_relay=False`) severs exactly this edge set, which provably reduces the
        # summary span to zero document content -- its only other keys are earlier summary spans, which
        # are severed for the same reason. Hence there is no non-vacuous "no path" control here.
        sum_reads_own_cell = same_cell & is_sum_q & ~is_sum_k & bool(pattern.summary_relay)
        # BANDWIDTH GATE: only the leading `summary_bandwidth` tokens of an earlier span are visible.
        # `summary_bandwidth=0` removes every cross-cell edge -> a pure cell-blocks mask.
        offset_kv = chunk_token_offset(chunk_ids).unsqueeze(1)  # (B, 1, S)
        visible_sum = is_sum_k & both_ctx & (offset_kv < pattern.summary_bandwidth)
        reads_earlier_sum = (cell_k < cell_q) & visible_sum
        context_ok = same_chunk | (
            (doc_reads_own_cell | sum_reads_own_cell | reads_earlier_sum) & (qc >= kc)
        )
    elif name == "gold_hop_controlled":
        # The ONE gold-aware pattern. Structurally identical to ``random_doc`` -- own document plus a
        # subset of the strictly-earlier ones -- except the subset arrives as an explicit per-example
        # doc->doc graph instead of being hashed from the indices, because it has been EDITED: the gold
        # pair's direct edge is deleted and a path of a controlled length forced in its place. Building
        # the graph needs the example's gold pairs, which must never touch the token stream, so it is
        # built in a fingerprint-keyed forward pre-hook and handed in here already finished. See
        # :mod:`olmo_core.nn.attention.gold_hop_mask`.
        if doc_adjacency is None:
            raise ValueError(
                "gold_hop_controlled requires a doc_adjacency tensor (the per-example, gold-edited "
                "doc->doc graph). Install the hook with gold_hop_mask.install_gold_hop_mask(); "
                "without it there is no graph and the arm would be silently wrong rather than absent."
            )
        if doc_adjacency.dim() == 2:
            doc_adjacency = doc_adjacency.unsqueeze(0)
        if doc_adjacency.shape[0] == 1 and B > 1:
            doc_adjacency = doc_adjacency.expand(B, -1, -1)
        n_d = doc_adjacency.shape[-1]
        max_id = int(chunk_ids.max().item())
        if max_id >= n_d:
            raise ValueError(
                f"doc_adjacency covers {n_d} documents but chunk_ids reference document {max_id}."
            )
        cross_doc = (qc > kc) & (qc >= 0) & (kc >= 0)
        # Gather adj[b, qc, kc] -> (B, S, S). Negative roles are clamped to a valid index and then
        # discarded by ``cross_doc`` (FREE/PAD/SINK never take this branch), so the clamp cannot leak.
        flat_idx = qc.clamp(min=0).to(torch.int64) * n_d + kc.clamp(min=0).to(torch.int64)
        adj_ok = (
            doc_adjacency.reshape(B, n_d * n_d).gather(1, flat_idx.reshape(B, -1)).reshape(B, S, S)
        )
        context_ok = same_chunk | (cross_doc & adj_ok)
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

    if name == "random_doc":
        keep_prob = pattern.doc_keep_prob
        seed_c = int(pattern.random_seed) * _RD_C
        # Per-example nonces are precomputed here (once per forward) and closed over, so the mask_mod
        # body stays a pure elementwise function of (b, q_idx, kv_idx) -- Triton-friendly.
        nonces = random_doc_nonce(cids) if pattern.random_doc_per_example else None

        def mask_mod(b, h, q_idx, kv_idx):
            qc = cids[b, q_idx]
            kc = cids[b, kv_idx]
            q_np = qc != PAD_CHUNK_ID
            kv_np = kc != PAD_CHUNK_ID
            same = (qc == kc) & (qc >= 0)
            cross = (qc > kc) & (qc >= 0) & (kc >= 0)
            # Same multiplicative hash as _random_doc_keep (int64 wraps mod 2**64 on both torch and
            # Triton; only the low 31 bits are used, so wrapping is well-defined and identical).
            hh = (qc.to(torch.int64) * _RD_A) ^ (kc.to(torch.int64) * _RD_B) ^ seed_c
            if nonces is not None:
                hh = hh ^ nonces[b]
            hh = (hh * _RD_MIX) & _RD_MASK
            keep = (hh.to(torch.float32) * (1.0 / float(_RD_MASK))) < keep_prob
            q_free = (qc < 0) & q_np
            kv_free = (kc < 0) & kv_np
            return (
                (q_idx >= kv_idx) & q_np & kv_np & (same | (cross & keep) | q_free | kv_free)
            ) | (q_idx == kv_idx)

        return mask_mod

    if name == "summary_attention":
        p = pattern.summary_every_k + 1
        k_cell = pattern.summary_every_k
        bandwidth = pattern.summary_bandwidth
        relay = bool(pattern.summary_relay)
        # Precomputed once per forward and closed over, exactly like `random_doc`'s nonces, so the body
        # stays a pure elementwise function of (b, q_idx, kv_idx) -- Triton-friendly.
        offsets = chunk_token_offset(cids)

        def mask_mod(b, h, q_idx, kv_idx):
            qc = cids[b, q_idx]
            kc = cids[b, kv_idx]
            q_np = qc != PAD_CHUNK_ID
            kv_np = kc != PAD_CHUNK_ID
            same = (qc == kc) & (qc >= 0)
            both_ctx = (qc >= 0) & (kc >= 0)
            cell_q = qc // p
            cell_k = kc // p
            is_sum_q = (qc % p) == k_cell
            is_sum_k = (kc % p) == k_cell
            same_cell = (cell_q == cell_k) & both_ctx
            doc_own = same_cell & ~is_sum_q & ~is_sum_k
            sum_own = same_cell & is_sum_q & ~is_sum_k & relay
            visible_sum = is_sum_k & both_ctx & (offsets[b, kv_idx] < bandwidth)
            earlier_sum = (cell_k < cell_q) & visible_sum
            ctx_ok = same | ((doc_own | sum_own | earlier_sum) & (qc >= kc))
            q_free = (qc < 0) & q_np
            kv_free = (kc < 0) & kv_np
            return ((q_idx >= kv_idx) & q_np & kv_np & (ctx_ok | q_free | kv_free)) | (
                q_idx == kv_idx
            )

        return mask_mod

    # ``gold_hop_controlled`` deliberately lands here: its graph comes from a per-forward Python hook
    # (fingerprint -> gold pairs -> edited graph), which is not torch.compile-capturable, so the whole
    # family runs eager anyway. Declining a mask_mod keeps it on the one dense boolean path rather than
    # splitting the arm across two code paths.
    return None  # unsupported pattern -> caller uses the dense materialized mask
