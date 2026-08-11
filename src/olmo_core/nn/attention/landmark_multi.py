"""
``MultiLandmarkAttention`` / ``DocumentMultiLandmarkAttention`` -- *multiple landmark tokens per
block* variants of the dense grouped-softmax landmark attention.

This module is **opt-in and self-contained**: it does not modify (or import with side effects) the
original single-landmark :class:`~olmo_core.nn.attention.LandmarkAttention` /
:class:`~olmo_core.nn.attention.DocumentLandmarkAttention` or
:func:`~olmo_core.nn.attention.landmark.landmark_grouped_softmax`, so existing landmark runs are
unaffected. The new classes subclass the originals and override only the mask construction and the
eager grouped-softmax call.

Layout: a block is ``block_size = mem_freq + num_landmarks`` tokens; the **last ``num_landmarks``
tokens of every block are landmarks** (``num_landmarks == 1`` reproduces the original layout, where
the single landmark sits at ``pos % block_size == block_size - 1``). The across-block ("memory")
softmax sees *all* of a block's landmarks, and the block's **gate** -- the weight by which the
block's regular tokens are scaled -- is a pool over its landmark probabilities, selected by
``landmark_pool``:

  * ``"sum"``: ``gate_b = sum_l p_{b,l}`` -- the marginal mass on the block's landmarks (equivalently
    a soft-OR / log-sum-exp over the block's landmark logits). Rows still sum to 1; gradients flow to
    every landmark of the block. This is the natural multi-vector generalization and the default.
  * ``"max"``: ``gate_b = max_l p_{b,l}`` -- the block is represented by its single best-matching
    landmark (the ColBERT-MaxSim analogue). Implemented by demoting every non-argmax landmark logit
    to ``-inf`` *before* the sum pool, so it is a hard max (rows still sum to 1; gradient flows to the
    winning landmark, with ties shared).

Both pools reduce to the original single-landmark grouped softmax when ``num_landmarks == 1`` (with
one landmark per block, ``sum`` and ``max`` coincide).

Eager-only (no fused kernel): the fused landmark Triton kernel assumes one periodic landmark, so the
multi-landmark path always uses the autograd-differentiable eager grouped softmax.
"""

from typing import Optional, Tuple

import torch

from olmo_core.exceptions import OLMoConfigurationError

from . import DocumentLandmarkAttention, LandmarkAttention
from .chunked_mask import (
    CHUNKED_ATTENTION_PATTERNS,
    AttentionPattern,
    build_chunked_allowed_mask,
)
from .landmark import LandmarkGroupedSoftmaxFunction

__all__ = [
    "LANDMARK_POOLS",
    "multi_landmark_grouped_softmax",
    "MultiLandmarkAttention",
    "DocumentMultiLandmarkAttention",
]

LANDMARK_POOLS = ("sum", "max")


def _block_id(T: int, dim: int, ndim: int, block_size: int, device: torch.device) -> torch.Tensor:
    """Per-key absolute block index ``key_pos // block_size``, shaped to broadcast along ``dim``."""
    shape = [1] * ndim
    shape[dim] = T
    return (torch.arange(T, device=device) // block_size).view(shape)


def _demote_nonmax_landmark_logits(
    x: torch.Tensor, dim: int, is_mem: torch.Tensor, block_size: int
) -> torch.Tensor:
    """
    For ``landmark_pool="max"``: within each block keep only the highest-logit landmark key, demoting
    every other landmark logit to ``finfo.min``. After this, the sum pool over a block's landmarks
    equals the single surviving (argmax) landmark's probability -- a hard max. Ties (equal max logits)
    are all kept and shared by the sum.

    :param x: Attention logits, e.g. ``(B, n_heads, T, T)``.
    :param is_mem: Boolean active-landmark mask, broadcastable to ``x``.
    """
    T = x.shape[dim]
    n_blocks = (T + block_size - 1) // block_size
    finfo_min = torch.finfo(x.dtype).min
    neg = torch.full((), finfo_min, dtype=x.dtype, device=x.device)

    block_id = _block_id(T, dim, x.dim(), block_size, x.device)
    block_idx_full = block_id.expand_as(x)
    # Logits restricted to active landmark keys (non-landmarks -> finfo.min so they can't win the max).
    lm_logits = torch.where(is_mem, x, neg)
    max_shape = list(x.shape)
    max_shape[dim] = n_blocks
    block_max = x.new_full(max_shape, finfo_min)
    block_max.scatter_reduce_(dim, block_idx_full, lm_logits, reduce="amax", include_self=True)
    per_key_block_max = torch.gather(block_max, dim, block_idx_full)
    demote = is_mem & (x < per_key_block_max)
    return torch.where(demote, neg, x)


def multi_landmark_grouped_softmax(
    x: torch.Tensor,
    dim: int,
    is_mem: torch.Tensor,
    last_section_mask: torch.Tensor,
    block_size: int,
    pool: str = "sum",
) -> torch.Tensor:
    """
    Dense grouped (two-level) softmax with **multiple landmark tokens per block**.

    Keys are grouped by absolute block index (``key_pos // block_size``), so a block may carry more
    than one landmark. The across-block softmax runs over every visible landmark plus the query's own
    section; each block's gate pools its landmarks' across-block probabilities (see the module
    docstring for ``pool``); and each earlier block's regular tokens get an independent within-block
    softmax scaled by that gate.

    :param x: Attention logits, e.g. of shape ``(B, n_heads, T, T)``.
    :param dim: The dimension to normalize over (the key dimension).
    :param is_mem: Boolean mask (broadcastable to ``x``) marking active landmark key positions.
    :param last_section_mask: Boolean mask (broadcastable to ``x``) marking, for each query, the keys
        in its own ("last") section.
    :param block_size: The landmark block size (``mem_freq + num_landmarks``).
    :param pool: ``"sum"`` or ``"max"`` (see the module docstring).

    :raises ValueError: If ``pool`` is not one of :data:`LANDMARK_POOLS`.
    """
    if pool not in LANDMARK_POOLS:
        raise ValueError(f"Unknown landmark_pool {pool!r}; expected one of {LANDMARK_POOLS}")
    if pool == "max":
        x = _demote_nonmax_landmark_logits(x, dim, is_mem, block_size)

    T = x.shape[dim]
    n_blocks = (T + block_size - 1) // block_size
    # Bucket layout: 0..n_blocks-1 are the per-block (within-block) groups; ``top`` is the shared
    # across-block ("memory") group that holds every active landmark plus the query's own section.
    top = n_blocks
    block_id = _block_id(T, dim, x.dim(), block_size, x.device)
    top_t = torch.full((), top, dtype=block_id.dtype, device=x.device)

    full_access_mask = is_mem | last_section_mask
    resp_mem_idx = torch.where(full_access_mask, top_t, block_id)
    probs = LandmarkGroupedSoftmaxFunction.apply(x, dim, n_blocks + 1, resp_mem_idx)

    # Block gate = sum over the block's landmark across-block probabilities (scatter-ADD accumulates
    # the block's landmarks). Non-landmark keys dump a zero into the unused ``top`` slot.
    gate_shape = list(x.shape)
    gate_shape[dim] = n_blocks + 1
    group_prob = probs.new_zeros((*gate_shape,))
    gate_index = torch.where(is_mem, block_id, top_t).expand_as(probs)
    gate_src = torch.where(is_mem, probs, probs.new_zeros(()))
    group_prob.scatter_add_(dim, gate_index, gate_src)

    probs = probs.mul(
        torch.where(
            full_access_mask,
            last_section_mask.to(probs.dtype),
            torch.gather(group_prob, dim, resp_mem_idx),
        )
    )
    return probs


def _single_doc_masks(
    T: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    block_size: int,
    num_landmarks: int,
    cu_doc_lens: Optional[torch.Tensor],
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generalized copy of :meth:`LandmarkAttention._landmark_masks` whose periodic ``is_mem`` marks the
    last ``num_landmarks`` positions of each block (instead of just the final one). The rest -- causal
    mask, optional ``cu_doc_lens`` block-diagonal document masking, ``mem_ids`` / ``last_section_mask``
    -- is unchanged, since a block's landmarks are contiguous at its end.
    """
    finfo_min = torch.finfo(dtype).min

    causal = torch.full((T, T), finfo_min, dtype=dtype, device=device)
    attn_mask = torch.triu(causal, diagonal=1)[None, None].clone()

    if cu_doc_lens is not None:
        boundaries = cu_doc_lens.to(device=device, dtype=torch.long)
        if bool((boundaries % block_size != 0).any().item()):
            raise OLMoConfigurationError(
                f"Multi-landmark packing requires every document boundary to be a multiple of the "
                f"landmark block size (mem_freq + num_landmarks = {block_size}), but got "
                f"cu_doc_lens={cu_doc_lens.tolist()}."
            )
        total = batch_size * T
        if int(boundaries[-1].item()) != total:
            raise OLMoConfigurationError(
                f"cu_doc_lens must describe exactly batch_size * T = {total} tokens (flattened over "
                f"the batch), but its last entry is {int(boundaries[-1].item())}."
            )
        interior = boundaries[(boundaries > 0) & (boundaries < total)]
        positions = torch.arange(total, device=device)
        doc_id = torch.searchsorted(interior, positions, right=True).view(batch_size, T)
        cross_doc = doc_id[:, None, :, None] != doc_id[:, None, None, :]
        attn_mask = attn_mask.masked_fill(cross_doc, finfo_min)

    is_mem = ((torch.arange(T, device=device) % block_size) >= (block_size - num_landmarks)).view(
        1, 1, 1, T
    )
    mem_ids = torch.where(attn_mask < -1, -1, torch.cumsum(is_mem, -1) - is_mem.int())
    last_section_mask = torch.amax(mem_ids, -1, keepdim=True) == mem_ids
    attn_mask.masked_fill_(last_section_mask & is_mem, finfo_min)
    last_section_mask = last_section_mask.logical_and(attn_mask > -1)
    is_mem = is_mem.logical_and(attn_mask > -1)
    return attn_mask, is_mem, last_section_mask


def _chunked_masks(
    T: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    block_size: int,
    num_landmarks: int,
    chunk_ids: torch.Tensor,
    pattern: AttentionPattern,
    is_anchor: Optional[torch.Tensor],
    batch_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generalized copy of :meth:`DocumentLandmarkAttention._landmark_masks` (the ``chunk_ids`` branch)
    whose periodic ``is_mem`` marks the last ``num_landmarks`` positions of each block.
    """
    finfo_min = torch.finfo(dtype).min

    chunk_ids = chunk_ids.to(device=device)
    if chunk_ids.dim() == 1:
        chunk_ids = chunk_ids.unsqueeze(0)
    if chunk_ids.shape[-1] != T:
        raise OLMoConfigurationError(
            f"chunk_ids last dim ({chunk_ids.shape[-1]}) must equal the sequence length ({T})."
        )
    if chunk_ids.shape[0] == 1 and batch_size > 1:
        chunk_ids = chunk_ids.expand(batch_size, T)

    allowed = build_chunked_allowed_mask(pattern, chunk_ids, is_anchor=is_anchor)
    attn_mask = torch.where(
        allowed.unsqueeze(1),
        torch.zeros((), dtype=dtype, device=device),
        torch.full((), finfo_min, dtype=dtype, device=device),
    )

    is_mem = ((torch.arange(T, device=device) % block_size) >= (block_size - num_landmarks)).view(
        1, 1, 1, T
    )
    mem_ids = torch.where(attn_mask < -1, -1, torch.cumsum(is_mem, -1) - is_mem.int())
    last_section_mask = torch.amax(mem_ids, -1, keepdim=True) == mem_ids
    attn_mask.masked_fill_(last_section_mask & is_mem, finfo_min)
    last_section_mask = last_section_mask.logical_and(attn_mask > -1)
    is_mem = is_mem.logical_and(attn_mask > -1)
    return attn_mask, is_mem, last_section_mask


class MultiLandmarkAttention(LandmarkAttention):
    """
    Multi-landmark dense grouped-softmax attention (``AttentionType.multi_landmark``).

    Subclasses :class:`LandmarkAttention`; only the periodic ``is_mem`` pattern (last
    ``num_landmarks`` positions of each block) and the eager grouped softmax (which now pools each
    block's landmarks via ``landmark_pool``) change. ``num_landmarks == 1`` reproduces
    :class:`LandmarkAttention` exactly.

    :param mem_freq: Regular tokens per block; the block size is ``mem_freq + num_landmarks``.
    :param num_landmarks: Number of landmark tokens at the end of each block (``>= 1``).
    :param landmark_pool: How to pool a block's landmark probabilities into its gate; one of
        :data:`LANDMARK_POOLS` (``"sum"`` or ``"max"``). Defaults to ``"sum"``.

    See :class:`LandmarkAttention` / :class:`Attention` for the remaining parameters.
    """

    def __init__(
        self,
        *,
        mem_freq: int,
        num_landmarks: int = 1,
        landmark_pool: str = "sum",
        use_kernel: bool = False,
        softmax_scale: Optional[float] = None,
        **kwargs,
    ):
        if use_kernel:
            raise OLMoConfigurationError(
                "MultiLandmarkAttention does not support the fused landmark kernel (the multi-landmark "
                "grouped softmax is eager); use the eager path (use_kernel=False)."
            )
        if num_landmarks < 1:
            raise OLMoConfigurationError(f"num_landmarks must be >= 1 (got {num_landmarks})")
        if landmark_pool not in LANDMARK_POOLS:
            raise OLMoConfigurationError(
                f"Unknown landmark_pool {landmark_pool!r}; expected one of {LANDMARK_POOLS}"
            )
        super().__init__(mem_freq=mem_freq, use_kernel=False, softmax_scale=softmax_scale, **kwargs)
        self.num_landmarks = num_landmarks
        self.landmark_pool = landmark_pool
        # Override the single-landmark block size set by ``LandmarkAttention.__init__``.
        self.block_size = mem_freq + num_landmarks

    def _landmark_masks(
        self,
        T: int,
        device: torch.device,
        dtype: torch.dtype,
        cu_doc_lens: Optional[torch.Tensor] = None,
        batch_size: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _single_doc_masks(
            T,
            device,
            dtype,
            block_size=self.block_size,
            num_landmarks=self.num_landmarks,
            cu_doc_lens=cu_doc_lens,
            batch_size=batch_size,
        )

    def _eager_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_doc_lens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, n_heads, T, _ = q.shape
        attn_mask, is_mem, last_section_mask = self._landmark_masks(
            T, q.device, q.dtype, cu_doc_lens=cu_doc_lens, batch_size=B
        )
        attn = torch.matmul(q, k.transpose(-1, -2)) * self.softmax_scale
        attn = attn + attn_mask
        attn = torch.maximum(
            attn, torch.tensor(torch.finfo(attn.dtype).min, device=attn.device, dtype=attn.dtype)
        )
        probs = multi_landmark_grouped_softmax(
            attn,
            dim=-1,
            is_mem=is_mem.expand(B, n_heads, T, T),
            last_section_mask=last_section_mask.expand(B, 1, T, T),
            block_size=self.block_size,
            pool=self.landmark_pool,
        ).to(q.dtype)
        return torch.matmul(probs, v)


class DocumentMultiLandmarkAttention(DocumentLandmarkAttention):
    """
    Document-chunked multi-landmark attention (``AttentionType.document_multi_landmark``): the
    chunked-document masking of :class:`DocumentLandmarkAttention` combined with multiple landmark
    tokens per block (see :class:`MultiLandmarkAttention`). ``num_landmarks == 1`` reproduces
    :class:`DocumentLandmarkAttention` exactly.

    :param num_landmarks: Number of landmark tokens at the end of each block (``>= 1``).
    :param landmark_pool: Pooling for a block's landmark gate; one of :data:`LANDMARK_POOLS`.

    See :class:`DocumentLandmarkAttention` for the remaining parameters.
    """

    def __init__(
        self,
        *,
        mem_freq: int,
        num_landmarks: int = 1,
        landmark_pool: str = "sum",
        cross_doc_mode: str = "chunked",
        doc_window_k: int = 0,
        token_window_w: int = 0,
        use_kernel: bool = False,
        softmax_scale: Optional[float] = None,
        **kwargs,
    ):
        if num_landmarks < 1:
            raise OLMoConfigurationError(f"num_landmarks must be >= 1 (got {num_landmarks})")
        if landmark_pool not in LANDMARK_POOLS:
            raise OLMoConfigurationError(
                f"Unknown landmark_pool {landmark_pool!r}; expected one of {LANDMARK_POOLS}"
            )
        if cross_doc_mode not in CHUNKED_ATTENTION_PATTERNS:
            raise OLMoConfigurationError(
                f"Unknown cross_doc_mode {cross_doc_mode!r}; expected one of "
                f"{CHUNKED_ATTENTION_PATTERNS}"
            )
        super().__init__(
            mem_freq=mem_freq,
            cross_doc_mode=cross_doc_mode,
            doc_window_k=doc_window_k,
            token_window_w=token_window_w,
            use_kernel=use_kernel,
            softmax_scale=softmax_scale,
            **kwargs,
        )
        self.num_landmarks = num_landmarks
        self.landmark_pool = landmark_pool
        self.block_size = mem_freq + num_landmarks

    def _landmark_masks(
        self,
        T: int,
        device: torch.device,
        dtype: torch.dtype,
        cu_doc_lens: Optional[torch.Tensor] = None,
        batch_size: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        chunk_ids = self._chunk_ids
        if chunk_ids is None:
            return _single_doc_masks(
                T,
                device,
                dtype,
                block_size=self.block_size,
                num_landmarks=self.num_landmarks,
                cu_doc_lens=cu_doc_lens,
                batch_size=batch_size,
            )
        is_anchor = self._build_is_anchor(chunk_ids) if self._pattern.needs_anchor() else None
        return _chunked_masks(
            T,
            device,
            dtype,
            block_size=self.block_size,
            num_landmarks=self.num_landmarks,
            chunk_ids=chunk_ids,
            pattern=self._pattern,
            is_anchor=is_anchor,
            batch_size=batch_size,
        )

    _eager_forward = MultiLandmarkAttention._eager_forward
