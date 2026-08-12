"""
:class:`SummaryTokenAttention` -- attention restricted by the per-document **summary-token** mask
(see :mod:`olmo_core.nn.attention.summary_mask` for the rule and its levers).

Structurally this mirrors :class:`~olmo_core.nn.attention.document_chunked.DocumentChunkedAttention`
-- the projections, QK-norm, RoPE, output gate and output projection are all inherited, and only
:meth:`sdpa` is overridden -- but it differs in two ways that matter at long context:

**The block mask is built analytically, not by evaluating the predicate.**
``torch.nn.attention.flex_attention.create_block_mask`` calls ``create_mask``, which nested-``vmap``s
the ``mask_mod`` over all four index dimensions and therefore materializes *every intermediate* at
full ``(B, H, T, T)``. Measured on CPU with this predicate that is ~3 GiB and ~4 s at ``T=16384``,
growing quadratically to roughly **760 GiB and 17 minutes at T=262144** -- and it is rebuilt every
microbatch forward. :func:`build_summary_block_mask` instead reduces the roles to a handful of
per-block statistics and derives the allowed block set from those directly, which is a
``(n_blocks, n_blocks)`` boolean -- 4 MB and milliseconds at 256k.

That is safe because :meth:`BlockMask.from_kv_blocks` re-applies ``mask_mod`` *inside* every partial
block, so a **superset** of the true block set changes sparsity but never numerics. Blocks are only
declared *full* (predicate skipped) under a strictly sufficient condition. The equivalence is pinned
in ``src/test/nn/attention/summary_token_attention_test.py``.

**GQA is native.** The flex kernel is given the unexpanded key/value heads with ``enable_gqa=True``
rather than materializing ``n_heads``-wide copies via ``_repeat_kv``, which on Qwen3.5-4B
(``n_heads=16``, ``n_kv_heads=4``) is a 4x saving on the two largest activation tensors.

No KV-caching, no intra-document packing (``cu_doc_lens``), and Ulysses context parallelism only.
"""

import logging
from typing import Optional

import torch
import torch.nn.functional as F

from olmo_core.distributed.parallel.context_parallel import (
    all_to_all_qkv_cp2hp,
    all_to_all_single_hp2cp,
)
from olmo_core.exceptions import OLMoConfigurationError

from . import Attention
from .backend import _repeat_kv
from .summary_mask import (
    ROLE_DOC_ID,
    ROLE_KIND,
    SummaryMaskSpec,
    TokenKind,
    build_summary_mask_mod,
    summary_mask_allowed,
)

log = logging.getLogger(__name__)

try:
    from torch.nn.attention.flex_attention import BlockMask, flex_attention

    _flex_attention = torch.compile(flex_attention)
    _HAS_FLEX = True
except Exception:  # pragma: no cover - flex unavailable on old torch
    _HAS_FLEX = False

#: FlexAttention only wins at long context; below this the materialized mask is faster.
_FLEX_MIN_SEQ_LEN = 8192

#: Granularity at which fully-masked blocks are skipped.
_DEFAULT_BLOCK_SIZE = 128

#: Above this, a dense ``(B, 1, T, T)`` additive mask is not a viable fallback (~2*B*T**2 bytes).
_DENSE_MASK_MAX_SEQ_LEN = 32768

#: Single-slot cache shared by every layer within one forward: the mask depends only on the roles
#: tensor (the same object threaded to every block), the sequence length, the spec and the block
#: size, all identical across layers.
_BLOCK_MASK_CACHE: dict = {}


def build_summary_block_mask(
    roles: torch.Tensor,
    spec: SummaryMaskSpec,
    *,
    causal_example: Optional[torch.Tensor] = None,
    block_size: int = _DEFAULT_BLOCK_SIZE,
    mask_mod=None,
):
    """
    Build a :class:`BlockMask` for the summary-token mask **without** evaluating the predicate
    pointwise.

    Each ``block_size``-token block is reduced to the document range it spans, the earliest document
    any of its summary tokens belongs to, and whether it holds instruction / query / padding tokens.
    A query block may reach a key block iff their document ranges overlap, or the key block holds a
    summary token of a document strictly earlier than some document in the query block, or it holds
    instruction text -- each a **conservative over-approximation** of the token-level rule, which is
    exactly what ``from_kv_blocks`` permits since it re-applies ``mask_mod`` within every partial
    block.

    Blocks are marked *full* (predicate skipped entirely) only when both are wholly document content
    of the same document, entirely free of padding, and the key block is strictly earlier -- under
    which every pair in the block is allowed regardless of the spec's flags.

    :param roles: ``(B, 3, T)`` from :func:`~olmo_core.nn.attention.summary_mask.build_summary_roles`.
    :param spec: The :class:`~olmo_core.nn.attention.summary_mask.SummaryMaskSpec`.
    :param causal_example: Optional ``(B,)`` bool marking examples in the causal arm.
    :param block_size: Query/key block size.
    :param mask_mod: The ``mask_mod`` to attach; built from ``roles``/``spec`` when omitted.

    :returns: A :class:`BlockMask`, or ``None`` if the sequence length is not a multiple of
        ``block_size`` (the caller falls back).
    """
    doc_id = roles[:, ROLE_DOC_ID]
    kind = roles[:, ROLE_KIND]
    B, T = kind.shape
    if T % block_size != 0:
        return None
    nb = T // block_size
    device = kind.device

    d = doc_id.view(B, nb, block_size).to(torch.int64)
    k = kind.view(B, nb, block_size)

    big = torch.iinfo(torch.int32).max
    belongs = d >= 0
    min_doc = torch.where(belongs, d, torch.full_like(d, big)).amin(-1)  # (B, nb)
    max_doc = torch.where(belongs, d, torch.full_like(d, -1)).amax(-1)

    is_summary = k == int(TokenKind.SUMMARY)
    min_summary_doc = torch.where(is_summary, d, torch.full_like(d, big)).amin(-1)

    has_instruction = (k == int(TokenKind.INSTRUCTION)).any(-1)
    has_query = (k == int(TokenKind.QUERY)).any(-1)
    not_pad = k != int(TokenKind.PAD)
    has_content = not_pad.any(-1)
    all_content = (k == int(TokenKind.DOC_CONTENT)).all(-1)

    blk = torch.arange(nb, device=device)
    causal_blk = (blk.view(-1, 1) >= blk.view(1, -1)).unsqueeze(0)  # (1, nb, nb)

    q_min, q_max = min_doc.unsqueeze(2), max_doc.unsqueeze(2)
    k_min, k_max = min_doc.unsqueeze(1), max_doc.unsqueeze(1)

    shares_document = (q_min <= k_max) & (k_min <= q_max) & (q_max >= 0) & (k_max >= 0)
    holds_earlier_summary = min_summary_doc.unsqueeze(1) < q_max
    holds_instruction = has_instruction.unsqueeze(1)

    reachable = shares_document | holds_earlier_summary | holds_instruction
    if spec.query_reads_documents:
        reachable = reachable | has_query.unsqueeze(2)

    if causal_example is None:
        ce = torch.zeros(B, 1, 1, dtype=torch.bool, device=device)
    else:
        ce = causal_example.to(device=device, dtype=torch.bool).reshape(B, 1, 1)

    allowed = causal_blk & has_content.unsqueeze(2) & has_content.unsqueeze(1) & (ce | reachable)
    # The self-diagonal NaN guard lives inside the diagonal block, so it must never be dropped.
    allowed = allowed | torch.eye(nb, dtype=torch.bool, device=device).unsqueeze(0)

    strictly_earlier = (blk.view(-1, 1) > blk.view(1, -1)).unsqueeze(0)
    single_doc = (min_doc == max_doc) & (max_doc >= 0)
    whole_doc_block = all_content & single_doc & not_pad.all(-1)
    full = (
        strictly_earlier
        & whole_doc_block.unsqueeze(2)
        & whole_doc_block.unsqueeze(1)
        & (max_doc.unsqueeze(2) == max_doc.unsqueeze(1))
    )
    partial = allowed & ~full

    def _to_blocks(sel: torch.Tensor):
        num = sel.sum(-1).to(torch.int32).unsqueeze(1)  # (B, 1, nb)
        order = sel.to(torch.int8).argsort(dim=-1, descending=True, stable=True)
        return num, order.to(torch.int32).unsqueeze(1)  # (B, 1, nb, nb)

    kv_num_blocks, kv_indices = _to_blocks(partial)
    full_kv_num_blocks, full_kv_indices = _to_blocks(full)

    if mask_mod is None:
        mask_mod = build_summary_mask_mod(roles, spec, causal_example=causal_example)

    return BlockMask.from_kv_blocks(
        kv_num_blocks,
        kv_indices,
        full_kv_num_blocks,
        full_kv_indices,
        BLOCK_SIZE=(block_size, block_size),
        mask_mod=mask_mod,
        seq_lengths=(T, T),
    )


class SummaryTokenAttention(Attention):
    """
    Attention masked by the per-document summary-token rule
    (``AttentionType.summary_token``). See the module docstring.

    The per-token ``summary_roles`` and per-example ``causal_example`` tensors are threaded through
    the model forward and stashed by :meth:`forward` for the overridden :meth:`sdpa` to read. When
    ``summary_roles`` is absent this falls back to ordinary causal attention, so the variant is a
    strict superset of :class:`Attention`.

    :param n_summary_tokens: Summary tokens per document; must match the tokenized data.
    :param summary_visible_tokens: How many leading tokens of each summary run later documents may
        read. ``None`` (default) is all of them.
    :param summaries_read_own_document: Whether a summary run may read its own document (``False``
        is the information-free placebo).
    :param summaries_read_earlier_summaries: Whether summary runs relay from earlier summary runs.
    :param query_reads_documents: Whether the trailing query/answer is an unrestricted reader.
    :param flex_block_size: FlexAttention block size; the granularity of block skipping.

    See :class:`Attention` for the remaining parameters.
    """

    def __init__(
        self,
        *,
        n_summary_tokens: int = 5,
        summary_visible_tokens: Optional[int] = None,
        summaries_read_own_document: bool = True,
        summaries_read_earlier_summaries: bool = True,
        query_reads_documents: bool = False,
        flex_block_size: int = _DEFAULT_BLOCK_SIZE,
        softmax_scale: Optional[float] = None,
        **kwargs,
    ):
        if kwargs.get("window_size") is not None:
            raise OLMoConfigurationError(
                "SummaryTokenAttention does not support sliding-window attention (the summary mask "
                "already governs which keys a query may see)."
            )
        self._dropout_p = float(kwargs.get("dropout") or 0.0)
        super().__init__(softmax_scale=softmax_scale, **kwargs)
        self.spec = SummaryMaskSpec(
            n_summary_tokens=n_summary_tokens,
            summary_visible_tokens=summary_visible_tokens,
            summaries_read_own_document=summaries_read_own_document,
            summaries_read_earlier_summaries=summaries_read_earlier_summaries,
            query_reads_documents=query_reads_documents,
        )
        self.flex_block_size = int(flex_block_size)
        self.softmax_scale = softmax_scale if softmax_scale is not None else self.head_dim**-0.5
        # Transient per-forward state, stashed by ``forward`` for ``sdpa`` to read.
        self._summary_roles: Optional[torch.Tensor] = None
        self._causal_example: Optional[torch.Tensor] = None
        # Sticky fallback: set if FlexAttention errors at runtime. Also settable by tests to force
        # the materialized path for flex-vs-dense parity checks.
        self._force_eager_mask: bool = False

    def forward(
        self,
        x: torch.Tensor,
        summary_roles: Optional[torch.Tensor] = None,
        causal_example: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Apply summary-token-masked attention.

        :param x: The input, ``(B, T, d_model)``.
        :param summary_roles: ``(B, 3, T)`` per-token roles; see
            :func:`~olmo_core.nn.attention.summary_mask.build_summary_roles`.
        :param causal_example: ``(B,)`` bool marking examples in the causal arm of the mixture.
        """
        self._summary_roles = summary_roles
        self._causal_example = causal_example
        try:
            return super().forward(x, **kwargs)
        finally:
            self._summary_roles = None
            self._causal_example = None

    def sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_doc_lens: Optional[torch.Tensor] = None,
        cu_doc_lens_q: Optional[torch.Tensor] = None,
        cu_doc_lens_k: Optional[torch.Tensor] = None,
        max_doc_len: Optional[int] = None,
        max_doc_len_q: Optional[int] = None,
        max_doc_len_k: Optional[int] = None,
        local_k_slice: Optional[slice] = None,
        cache_leftpad: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Attention over the full sequence restricted by the summary-token mask. ``q``/``k``/``v``
        arrive as ``(B, T, H, D)`` / ``(B, T, H_kv, D)``; the result is ``(B, T, H, D)``.
        """
        del max_doc_len, max_doc_len_q, max_doc_len_k, cache_leftpad
        if any(o is not None for o in (cu_doc_lens, cu_doc_lens_q, cu_doc_lens_k, local_k_slice)):
            raise NotImplementedError(
                "SummaryTokenAttention does not support intra-document packing (cu_doc_lens). The "
                "summary mask already governs document isolation for the layers it covers; set "
                "generate_doc_lengths=False on the data loader."
            )
        if self.kv_cache_manager is not None:
            raise NotImplementedError("SummaryTokenAttention does not support KV caching.")

        if not self.cp_enabled:
            return self._sdpa_masked(q, k, v)

        # ---- Ulysses context parallelism ----
        # Overriding ``sdpa`` bypasses the backend, so the all-to-all that Ulysses normally performs
        # there never happens. Without doing it here, q/k/v would still be sequence-sharded while the
        # roles are full-length, and the mask would be built for the wrong sequence -- silently, since
        # the shapes only disagree by the CP degree. After the gather, q/k/v hold the FULL sequence
        # with partitioned heads, which is exactly the object the mask describes.
        if getattr(self.backend, "uly", None) is None:
            raise OLMoConfigurationError(
                "SummaryTokenAttention supports Ulysses context parallelism only. Ring/zigzag CP "
                "permutes each rank's rows non-contiguously and its kernel understands only "
                "'causal + cu_seqlens', which cannot express this mask."
            )
        cp_pg = self.backend.cp_pg
        q, k, v = all_to_all_qkv_cp2hp(q, k, v, cp_pg)
        out = self._sdpa_masked(q, k, v)
        # ``_sdpa_masked`` already returns a contiguous (B, T, H/CP, D); the collective needs that.
        return all_to_all_single_hp2cp(out.contiguous(), cp_pg)

    def _prepared_roles(self, T: int, device: torch.device, batch_size: int) -> torch.Tensor:
        roles = self._summary_roles
        assert roles is not None
        roles = roles.to(device=device)
        if roles.dim() == 2:
            roles = roles.unsqueeze(0)
        if roles.shape[-1] != T:
            raise OLMoConfigurationError(
                f"summary_roles last dim ({roles.shape[-1]}) must equal the sequence length ({T}). "
                "Under context parallelism the roles must stay unsharded and full-length."
            )
        if roles.shape[0] == 1 and batch_size > 1:
            roles = roles.expand(batch_size, -1, -1)
        return roles

    def _block_mask(self, roles: torch.Tensor, B: int, T: int):
        """Build (or reuse) this forward's :class:`BlockMask`. Single-slot cache across layers."""
        src = self._summary_roles
        assert src is not None
        key = (
            id(src),
            src.data_ptr(),
            src._version,
            B,
            T,
            self.flex_block_size,
            self.spec.cache_key,
            None if self._causal_example is None else self._causal_example.data_ptr(),
            str(roles.device),
        )
        cached = _BLOCK_MASK_CACHE.get("cur")
        if cached is not None and cached[0] == key:
            return cached[1]
        block_mask = build_summary_block_mask(
            roles,
            self.spec,
            causal_example=self._causal_example,
            block_size=self.flex_block_size,
        )
        _BLOCK_MASK_CACHE["cur"] = (key, block_mask)
        return block_mask

    def _sdpa_masked(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, T, n_heads, _ = q.shape

        if self._summary_roles is None:
            # No roles -> plain causal attention (GQA expanded, as the torch backend does).
            n_rep = n_heads // k.shape[2]
            out = F.scaled_dot_product_attention(
                q.transpose(1, 2),
                _repeat_kv(k, n_rep).transpose(1, 2),
                _repeat_kv(v, n_rep).transpose(1, 2),
                dropout_p=self._dropout_p if self.training else 0.0,
                is_causal=True,
                scale=self.softmax_scale,
            )
            return out.transpose(1, 2).contiguous()

        roles = self._prepared_roles(T, q.device, B)

        if _HAS_FLEX and q.is_cuda and T >= _FLEX_MIN_SEQ_LEN and not self._force_eager_mask:
            out = self._run_flex(q, k, v, roles, B=B, T=T)
            if out is not None:
                return out
            if T > _DENSE_MASK_MAX_SEQ_LEN:
                gib = 2 * B * T * T / 2**30
                raise RuntimeError(
                    f"SummaryTokenAttention: FlexAttention could not run at seq_len={T} and the "
                    f"dense fallback mask would need ~{gib:.0f} GiB (B={B}). The dense T x T "
                    f"fallback is only viable at seq_len <= {_DENSE_MASK_MAX_SEQ_LEN}."
                )

        # Fallback: dense materialized additive mask (CPU/tests, short context, or a flex error).
        n_rep = n_heads // k.shape[2]
        allowed = summary_mask_allowed(roles, self.spec, causal_example=self._causal_example)
        bias = torch.where(
            allowed.unsqueeze(1),
            torch.zeros((), dtype=q.dtype, device=q.device),
            torch.full((), torch.finfo(q.dtype).min, dtype=q.dtype, device=q.device),
        )
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            _repeat_kv(k, n_rep).transpose(1, 2),
            _repeat_kv(v, n_rep).transpose(1, 2),
            attn_mask=bias,
            dropout_p=self._dropout_p if self.training else 0.0,
            is_causal=False,  # the mask already encodes causality
            scale=self.softmax_scale,
        )
        return out.transpose(1, 2).contiguous()

    def _run_flex(self, q, k, v, roles, *, B: int, T: int) -> Optional[torch.Tensor]:
        """Run the block-sparse path once. Returns ``None`` if flex could not run."""
        for attempt in range(2):
            try:
                block_mask = self._block_mask(roles, B, T)
                if block_mask is None:  # sequence length not a multiple of the block size
                    return None
                out = _flex_attention(
                    q.transpose(1, 2).contiguous(),
                    k.transpose(1, 2).contiguous(),
                    v.transpose(1, 2).contiguous(),
                    block_mask=block_mask,
                    scale=self.softmax_scale,
                    enable_gqa=k.shape[2] != q.shape[2],
                )
                return out.transpose(1, 2).contiguous()
            except torch.cuda.OutOfMemoryError:  # pragma: no cover - CUDA-only long-context path
                _BLOCK_MASK_CACHE.pop("cur", None)
                torch.cuda.empty_cache()
                if attempt == 1:
                    self._force_eager_mask = True
                    log.warning(
                        f"FlexAttention OOM at seq_len={T} even after empty_cache; giving up on the "
                        "block-sparse path for this SummaryTokenAttention forward."
                    )
                    return None
            except Exception as e:  # pragma: no cover - fall back if flex fails at runtime
                self._force_eager_mask = True
                log.warning(
                    f"FlexAttention failed ({e}); falling back to the dense summary mask for this "
                    "SummaryTokenAttention layer."
                )
                return None
        return None
