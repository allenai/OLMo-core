"""
Per-document **summary-token** attention masking (the ``SummTokenSFT`` experiment).

Each context document in an SFT example is followed by a short run of ``<|summ|>`` tokens -- its
*summary run*. On a **masked** example a document may read only itself plus the summary runs of
strictly earlier documents, and the trailing query/answer may read only the summary runs -- never raw
document content. On a **causal** example the mask degenerates to plain causal. Which examples are
causal is decided per forward by the mask-mix schedule (see
:func:`~olmo_core.nn.attention.chunked_mask.mask_mix_standard_prob`) and arrives here as an explicit
``causal_example`` flag.

This module deliberately does **not** reuse the ``"summary_attention"``
:class:`~olmo_core.nn.attention.chunked_mask.AttentionPattern`. That pattern expresses a different
experiment -- documents grouped into *cells* of ``summary_every_k``, with a relay *bandwidth* -- and
reaching this design through it requires encoding tricks (``summary_every_k=1``,
``summary_bandwidth=N``) whose meaning is not the thing being varied. It also packs the document index
and the token role into a single ``int32`` with negative sentinels, which makes the span test
``(chunk_id % (k+1)) == k`` alias on those sentinels (``-1 % (k+1) == k`` for every ``k``). Here the
document index and the role are **separate named fields**, so the rule can be written as the sentence
it is:

.. code-block:: text

    allowed = causal & neither-is-pad & (
          this example is causal
        | same document
        | the key is instruction text
        | the key is a summary token of an earlier document
        | the query is unrestricted (optional)
    )

:func:`summary_mask_allowed` (dense) and :func:`build_summary_mask_mod` (FlexAttention) are two
renderings of that one rule and are asserted element-identical in
``src/test/nn/attention/summary_mask_test.py``.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import torch

__all__ = [
    "TokenKind",
    "SummaryMaskSpec",
    "ROLE_DOC_ID",
    "ROLE_KIND",
    "ROLE_SUMMARY_OFFSET",
    "build_summary_roles",
    "summary_mask_allowed",
    "build_summary_mask_mod",
]


class TokenKind(IntEnum):
    """
    The role of a single token in a summary-token SFT example.

    Unlike the chunked-mask role convention, this is a *separate* field from the document index, so
    no value here doubles as a sentinel and no arithmetic on it can alias.
    """

    #: Padding. Never attends and is never attended (bar the self-diagonal NaN guard).
    PAD = 0
    #: The free instruction prefix. Readable by every non-pad token.
    INSTRUCTION = 1
    #: The body of a context document.
    DOC_CONTENT = 2
    #: One of the ``n_summary_tokens`` summary tokens appended to a context document.
    SUMMARY = 3
    #: The trailing query and answer.
    QUERY = 4


@dataclass
class SummaryMaskSpec:
    """
    The levers of the summary-token mask. Every field is something an arm actually varies, and the
    defaults are the *treatment* -- there is no setting that silently severs the summary edges.

    :param n_summary_tokens: How many summary tokens follow each context document. Must match the
        tokenized data; the attention layer asserts this against the shard metadata.
    :param summary_visible_tokens: How many *leading* tokens of each summary run a later document may
        read. ``None`` (the default) means all of them. Lowering it is the compression-dose ladder:
        the tokenized data is unchanged across rungs, only visibility varies. ``0`` removes the relay
        entirely and is the floor control -- reachable only by asking for it explicitly.
    :param summaries_read_own_document: Whether a document's summary tokens may read that document's
        content. ``True`` (default) is the treatment -- the run aggregates its document and carries it
        forward. ``False`` is the **placebo**: the run keeps its position, its tokens and every edge
        *into* it, but reads nothing, so it provably carries zero document content.
    :param summaries_read_earlier_summaries: Whether a summary run may read the summary runs of
        earlier documents. ``True`` (default) gives a relay chain, so information can propagate
        transitively past its immediate neighbour. ``False`` limits every document to summaries
        written directly from raw content.
    :param query_reads_documents: Whether the trailing query/answer is an unrestricted reader.
        ``False`` (default) is the treatment -- the query sees the instruction, its own span and the
        summary runs, but no raw document content. ``True`` makes the query global, restricting only
        document-to-document attention.
    """

    n_summary_tokens: int = 5
    summary_visible_tokens: Optional[int] = None
    summaries_read_own_document: bool = True
    summaries_read_earlier_summaries: bool = True
    query_reads_documents: bool = False

    def __post_init__(self) -> None:
        if self.n_summary_tokens < 1:
            raise ValueError(f"n_summary_tokens must be >= 1 (got {self.n_summary_tokens})")
        if self.summary_visible_tokens is not None and self.summary_visible_tokens < 0:
            raise ValueError(
                f"summary_visible_tokens must be >= 0 or None (got {self.summary_visible_tokens})"
            )

    @property
    def cache_key(self) -> tuple:
        """A hashable summary of every field that changes the mask, for the block-mask cache."""
        return (
            self.n_summary_tokens,
            self.summary_visible_tokens,
            self.summaries_read_own_document,
            self.summaries_read_earlier_summaries,
            self.query_reads_documents,
        )


#: Index of each field along dim 1 of the ``(B, 3, T)`` roles tensor returned by
#: :func:`build_summary_roles`.
ROLE_DOC_ID = 0
ROLE_KIND = 1
ROLE_SUMMARY_OFFSET = 2


def build_summary_roles(
    input_ids: torch.Tensor,
    *,
    doc_start_id: int,
    doc_end_id: int,
    summary_token_id: int,
    eos_id: int,
    pad_id: Optional[int] = None,
) -> torch.Tensor:
    """
    Derive per-token ``(doc_id, kind, summary_offset)`` from the token stream, vectorized.

    ``doc_id`` is the count of **completed summary runs** strictly before a position. A summary run
    terminates its document, so a document's content *and* its own summary run share that document's
    index, and everything after the last run -- the query/answer -- lands on ``n_docs``, one past the
    last document. That makes "an earlier document" a plain integer comparison with no special case,
    and it does not depend on documents and summary runs alternating in the token stream.

    ``kind`` is a :class:`TokenKind`. ``summary_offset`` is a token's position within its own summary
    run (``0`` for the first) and is meaningless -- but harmless -- elsewhere; it is precomputed here
    so the FlexAttention ``mask_mod`` body stays a pure elementwise function of its indices.

    :param input_ids: Token ids, shape ``(B, T)`` (or ``(T,)``).
    :param doc_start_id: The ``<|box_start|>`` token id.
    :param doc_end_id: The ``<|box_end|>`` token id.
    :param summary_token_id: The ``<|summ|>`` token id.
    :param eos_id: The EOS / example terminator. Everything strictly after the first one is padding.
    :param pad_id: Optional dedicated padding id. When given, those positions are ``PAD`` too, which
        matters whenever the tokenizer ties ``pad`` to ``eos`` (Qwen3.5 does).

    :returns: An int32 tensor of shape ``(B, 3, T)``; see :data:`ROLE_DOC_ID`, :data:`ROLE_KIND`,
        :data:`ROLE_SUMMARY_OFFSET`.
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    B, T = input_ids.shape
    device = input_ids.device
    pos = torch.arange(T, device=device).expand(B, T)

    is_summary = input_ids == summary_token_id

    # ---- doc_id: how many summary runs have *finished* before this position -------------------
    # A run's last token is a summary token whose successor is not one (the final position counts).
    next_is_summary = torch.zeros_like(is_summary)
    next_is_summary[:, :-1] = is_summary[:, 1:]
    run_end = is_summary & ~next_is_summary
    run_end_l = run_end.to(torch.long)
    # Exclusive cumsum: runs finished strictly before this position.
    doc_id = torch.cumsum(run_end_l, dim=1) - run_end_l
    n_docs = run_end_l.sum(dim=1, keepdim=True)  # (B, 1)

    # ---- summary_offset: position within the current summary run ------------------------------
    # Runs are maximal spans of equal ``is_summary``; the offset is pos - (first pos of this run).
    run_start = torch.ones_like(is_summary)
    run_start[:, 1:] = is_summary[:, 1:] != is_summary[:, :-1]
    first_pos = torch.cummax(torch.where(run_start, pos, torch.zeros_like(pos)), dim=1).values
    summary_offset = torch.where(is_summary, pos - first_pos, torch.zeros_like(pos))

    # ---- kind ---------------------------------------------------------------------------------
    # "Inside a box span" mirrors ``build_chunk_ids_from_tokens``: more spans opened than closed
    # before this token, which keeps the closing marker attached to its span.
    starts = input_ids == doc_start_id
    ends = input_ids == doc_end_id
    n_start = torch.cumsum(starts.to(torch.long), dim=1)
    n_end = torch.cumsum(ends.to(torch.long), dim=1)
    inside_span = n_start > (n_end - ends.to(torch.long))

    # Anything at or past the last summary run belongs to the trailing query/answer region; anything
    # before it is either document content (inside a span) or instruction text (outside one). Note
    # ``n_docs > 0``: an example with no summary runs at all has no trailing region to speak of, and
    # without this guard every in-span token would satisfy ``doc_id >= n_docs`` and be called QUERY.
    trailing = (doc_id >= n_docs) & (n_docs > 0)
    kind = torch.where(
        trailing,
        torch.full_like(input_ids, int(TokenKind.QUERY), dtype=torch.int32),
        torch.where(
            inside_span,
            torch.full_like(input_ids, int(TokenKind.DOC_CONTENT), dtype=torch.int32),
            torch.full_like(input_ids, int(TokenKind.INSTRUCTION), dtype=torch.int32),
        ),
    )
    # Summary tokens are identified by id, not by span, so this works whether the data wraps each
    # summary run in its own boundary markers or emits it inline at the end of a document.
    kind = torch.where(is_summary, torch.full_like(kind, int(TokenKind.SUMMARY)), kind)

    # PAD: everything strictly after the first EOS, plus any dedicated pad id.
    is_eos = input_ids == eos_id
    eos_pos = torch.where(is_eos, pos, torch.full_like(pos, T))
    pad_from = eos_pos.min(dim=1, keepdim=True).values + 1
    is_pad = pos >= pad_from
    if pad_id is not None:
        is_pad = is_pad | (input_ids == pad_id)
    kind = torch.where(is_pad, torch.full_like(kind, int(TokenKind.PAD)), kind)

    # Instruction and padding belong to no document. -1 keeps them out of ``same_document`` without
    # needing a special case there (the instruction is reachable via its own term instead).
    doc_id = torch.where(
        (kind == int(TokenKind.INSTRUCTION)) | (kind == int(TokenKind.PAD)),
        torch.full_like(doc_id, -1),
        doc_id,
    )

    return torch.stack(
        [doc_id.to(torch.int32), kind.to(torch.int32), summary_offset.to(torch.int32)], dim=1
    )


def _causal_example_row(causal_example: Optional[torch.Tensor], B: int, device) -> torch.Tensor:
    """Normalize the per-example causal flag to a ``(B,)`` bool tensor (all-False when absent)."""
    if causal_example is None:
        return torch.zeros(B, dtype=torch.bool, device=device)
    ce = causal_example.to(device=device, dtype=torch.bool).reshape(-1)
    if ce.numel() == 1 and B > 1:
        ce = ce.expand(B)
    if ce.numel() != B:
        raise ValueError(f"causal_example has {ce.numel()} entries but the batch is {B}")
    return ce


def summary_mask_allowed(
    roles: torch.Tensor,
    spec: SummaryMaskSpec,
    *,
    causal_example: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Materialize the summary-token mask as a dense ``(B, T, T)`` boolean (``True`` = may attend).

    ``O(T^2)`` memory, so this is for tests, short sequences, and as the reference the FlexAttention
    ``mask_mod`` is checked against -- long-context training uses
    :class:`~olmo_core.nn.attention.summary_token.SummaryTokenAttention`, which never materializes it.

    :param roles: ``(B, 3, T)`` from :func:`build_summary_roles`.
    :param spec: The :class:`SummaryMaskSpec`.
    :param causal_example: Optional ``(B,)`` bool. Where ``True`` that example is in the causal arm
        and its mask is plain causal (padding still excluded). ``None`` means no example is.

    :returns: A boolean ``(B, T, T)`` tensor; dim 1 is the query, dim 2 the key.
    """
    doc_id = roles[:, ROLE_DOC_ID]
    kind = roles[:, ROLE_KIND]
    summary_offset = roles[:, ROLE_SUMMARY_OFFSET]
    B, T = kind.shape
    device = kind.device

    idx = torch.arange(T, device=device)
    causal = (idx.unsqueeze(1) >= idx.unsqueeze(0)).unsqueeze(0)  # (1, T, T)

    q_doc, k_doc = doc_id.unsqueeze(2), doc_id.unsqueeze(1)
    q_kind, k_kind = kind.unsqueeze(2), kind.unsqueeze(1)
    k_offset = summary_offset.unsqueeze(1)

    q_not_pad = q_kind != int(TokenKind.PAD)
    k_not_pad = k_kind != int(TokenKind.PAD)

    # -- the rule ------------------------------------------------------------------------------
    same_document = (q_doc == k_doc) & (q_doc >= 0)
    if not spec.summaries_read_own_document:
        same_document = same_document & ~(
            (q_kind == int(TokenKind.SUMMARY)) & (k_kind == int(TokenKind.DOC_CONTENT))
        )

    key_is_instruction = k_kind == int(TokenKind.INSTRUCTION)

    key_is_earlier_summary = (k_kind == int(TokenKind.SUMMARY)) & (k_doc < q_doc)
    if spec.summary_visible_tokens is not None:
        key_is_earlier_summary = key_is_earlier_summary & (k_offset < spec.summary_visible_tokens)
    if not spec.summaries_read_earlier_summaries:
        key_is_earlier_summary = key_is_earlier_summary & (q_kind != int(TokenKind.SUMMARY))

    reachable = same_document | key_is_instruction | key_is_earlier_summary
    if spec.query_reads_documents:
        reachable = reachable | (q_kind == int(TokenKind.QUERY))

    ce = _causal_example_row(causal_example, B, device).view(B, 1, 1)
    allowed = causal & q_not_pad & k_not_pad & (ce | reachable)

    # A fully-masked query row would make softmax produce NaN; always let a query attend itself.
    # Such rows are dropped by the loss mask anyway.
    return allowed | torch.eye(T, dtype=torch.bool, device=device).unsqueeze(0)


def build_summary_mask_mod(
    roles: torch.Tensor,
    spec: SummaryMaskSpec,
    *,
    causal_example: Optional[torch.Tensor] = None,
):
    """
    Build a FlexAttention ``mask_mod`` closure for the summary-token mask.

    Element-equivalent to :func:`summary_mask_allowed` (same rule, same self-diagonal guard), so a
    block-sparse kernel computes exactly the same masked softmax while skipping fully-masked blocks.
    Everything the body needs is precomputed and closed over, keeping it a pure elementwise function
    of ``(b, h, q_idx, kv_idx)``.

    :param roles: ``(B, 3, T)`` from :func:`build_summary_roles`, already on the target device.
    :param spec: The :class:`SummaryMaskSpec`.
    :param causal_example: Optional ``(B,)`` bool; see :func:`summary_mask_allowed`.

    :returns: ``mask_mod(b, h, q_idx, kv_idx) -> bool``.
    """
    doc_id = roles[:, ROLE_DOC_ID]
    kind = roles[:, ROLE_KIND]
    summary_offset = roles[:, ROLE_SUMMARY_OFFSET]
    ce = _causal_example_row(causal_example, kind.shape[0], kind.device)

    PAD = int(TokenKind.PAD)
    INSTRUCTION = int(TokenKind.INSTRUCTION)
    DOC_CONTENT = int(TokenKind.DOC_CONTENT)
    SUMMARY = int(TokenKind.SUMMARY)
    QUERY = int(TokenKind.QUERY)

    visible = spec.summary_visible_tokens
    own_doc_for_summaries = spec.summaries_read_own_document
    relay = spec.summaries_read_earlier_summaries
    free_query = spec.query_reads_documents

    def mask_mod(b, h, q_idx, kv_idx):
        q_doc = doc_id[b, q_idx]
        k_doc = doc_id[b, kv_idx]
        q_kind = kind[b, q_idx]
        k_kind = kind[b, kv_idx]

        q_not_pad = q_kind != PAD
        k_not_pad = k_kind != PAD

        same_document = (q_doc == k_doc) & (q_doc >= 0)
        if not own_doc_for_summaries:
            same_document = same_document & ~((q_kind == SUMMARY) & (k_kind == DOC_CONTENT))

        key_is_instruction = k_kind == INSTRUCTION

        key_is_earlier_summary = (k_kind == SUMMARY) & (k_doc < q_doc)
        if visible is not None:
            key_is_earlier_summary = key_is_earlier_summary & (summary_offset[b, kv_idx] < visible)
        if not relay:
            key_is_earlier_summary = key_is_earlier_summary & (q_kind != SUMMARY)

        reachable = same_document | key_is_instruction | key_is_earlier_summary
        if free_query:
            reachable = reachable | (q_kind == QUERY)

        return ((q_idx >= kv_idx) & q_not_pad & k_not_pad & (ce[b] | reachable)) | (q_idx == kv_idx)

    return mask_mod
