"""
Helper ops for Landmark Attention.

Landmark attention (`Mohtashami & Jaggi, 2023 <https://arxiv.org/abs/2305.16300>`_) inserts a
special "landmark" (a.k.a. "memory") token after every ``mem_freq`` regular tokens, dividing the
sequence into blocks of ``block_size = mem_freq + 1`` tokens where the last token of each block is
the landmark. Attention is computed with a grouped (two-level) softmax so that a query attends to a
block's tokens gated by the attention weight assigned to that block's landmark.

This module holds the framework-agnostic pieces (the grouped softmax and GQA head expansion). The
:class:`~olmo_core.nn.attention.LandmarkAttention` module itself lives in
:mod:`olmo_core.nn.attention` alongside the other attention variants, and the fused Triton training
kernel lives in :mod:`olmo_core.nn.attention.landmark_kernel`.
"""

from typing import Tuple

import torch

__all__ = [
    "repeat_kv",
    "landmark_grouped_softmax",
    "LandmarkGroupedSoftmaxFunction",
    "build_landmark_masks",
]


def build_landmark_masks(
    T: int, block_size: int, device: torch.device, dtype: torch.dtype
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build the additive causal attention mask, the landmark mask (``is_mem``), and the
    ``last_section_mask`` used by the eager (dense) landmark grouped-softmax path. All are
    batch-independent (built with leading dims of 1) and depend only on ``T`` and the landmark
    ``block_size`` (``= mem_freq + 1``).

    :param T: The (padded) sequence length; must be a multiple of ``block_size``.
    :param block_size: The landmark block size (the landmark token is the last token of each block).
    :param device: The device to build the masks on.
    :param dtype: The dtype of the additive attention mask.

    :returns: ``(attn_mask, is_mem, last_section_mask)`` where ``attn_mask`` is the additive causal
        mask of shape ``(1, 1, T, T)`` (with the query's own-section landmark also masked), ``is_mem``
        marks past landmark key positions, and ``last_section_mask`` marks, for each query, the keys
        in its own ("last") section. ``is_mem`` and ``last_section_mask`` are boolean.
    """
    finfo_min = torch.finfo(dtype).min

    # additive causal mask, shape (1, 1, T, T): 0 on/below the diagonal, -inf above.
    causal = torch.full((T, T), finfo_min, dtype=dtype, device=device)
    attn_mask = torch.triu(causal, diagonal=1)[None, None].clone()

    # shape: (1, 1, 1, T)
    is_mem = ((torch.arange(T, device=device) % block_size) == (block_size - 1)).view(1, 1, 1, T)
    mem_ids = torch.where(attn_mask < -1, -1, torch.cumsum(is_mem, -1) - is_mem.int())
    last_section_mask = torch.amax(mem_ids, -1, keepdim=True) == mem_ids
    # Mask landmark tokens that fall in the query's own (last) section.
    attn_mask.masked_fill_(last_section_mask & is_mem, finfo_min)
    last_section_mask = last_section_mask.logical_and(attn_mask > -1)
    is_mem = is_mem.logical_and(attn_mask > -1)

    return attn_mask, is_mem, last_section_mask


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Expand key/value heads for grouped-query attention:
    ``(B, n_kv_heads, T, head_dim) -> (B, n_kv_heads * n_rep, T, head_dim)``.
    """
    if n_rep == 1:
        return hidden_states
    bsz, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        bsz, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(bsz, num_key_value_heads * n_rep, slen, head_dim)


class LandmarkGroupedSoftmaxFunction(torch.autograd.Function):
    """
    Grouped softmax used by landmark attention. Adapted from the reference implementation.
    """

    @staticmethod
    def forward(ctx, x, dim, mem_cnt, resp_mem_idx):
        new_shape = list(x.shape)
        new_shape[dim] = mem_cnt
        max_by_group = x.new_zeros((*new_shape,))
        max_by_group.scatter_reduce_(
            src=x, index=resp_mem_idx, dim=dim, reduce="amax", include_self=False
        )

        maxes = torch.gather(max_by_group, dim, resp_mem_idx)
        x_exp = torch.exp((x - maxes).to(torch.float32))

        cumsum_by_group = torch.zeros_like(max_by_group, dtype=x_exp.dtype)
        cumsum_by_group.scatter_add_(dim, resp_mem_idx, x_exp)
        denom = torch.gather(cumsum_by_group, dim, resp_mem_idx)

        probs = x_exp / denom

        ctx.mem_cnt = mem_cnt
        ctx.dim = dim
        ctx.save_for_backward(resp_mem_idx, probs)

        return probs

    @staticmethod
    def backward(ctx, grad_probs):
        mem_cnt = ctx.mem_cnt
        dim = ctx.dim
        resp_mem_idx, probs = ctx.saved_tensors
        grad_x = grad_dim = grad_mem_cnt = grad_resp_mem_idx = None

        if ctx.needs_input_grad[0] or ctx.needs_input_grad[4]:
            grad_pair = grad_probs * probs

            new_shape = list(probs.shape)
            new_shape[dim] = mem_cnt
            cumsum_by_group = grad_pair.new_zeros((*new_shape,))
            cumsum_by_group.scatter_add_(dim, resp_mem_idx, grad_pair)

        if ctx.needs_input_grad[0]:
            grad_sum = torch.gather(cumsum_by_group, dim, resp_mem_idx)
            grad_x = grad_pair - probs * grad_sum
        assert not ctx.needs_input_grad[1]
        assert not ctx.needs_input_grad[2]
        assert not ctx.needs_input_grad[3]

        return grad_x, grad_dim, grad_mem_cnt, grad_resp_mem_idx


def landmark_grouped_softmax(
    x: torch.Tensor,
    dim: int,
    is_mem: torch.Tensor,
    last_section_mask: torch.Tensor,
) -> torch.Tensor:
    """
    The eager (dense) grouped softmax used by landmark attention.

    :param x: The attention logits, e.g. of shape ``(B, n_heads, T, T)``.
    :param dim: The dimension to normalize over (the key dimension).
    :param is_mem: Boolean mask (broadcastable to ``x``) marking landmark key positions.
    :param last_section_mask: Boolean mask (broadcastable to ``x``) marking, for each query, the
        keys that belong to the query's own ("last") section.
    """
    full_access_mask = is_mem | last_section_mask

    max_mem_cnt = int(is_mem.sum(dim=dim).max().item()) + 1
    mem_group_idx = torch.cumsum(is_mem, dim=dim)
    mem_bucket_id = max_mem_cnt - 1
    resp_mem_idx = torch.where(
        last_section_mask,
        max_mem_cnt - 1,
        torch.where(is_mem, mem_bucket_id, mem_group_idx),
    )
    probs = LandmarkGroupedSoftmaxFunction.apply(x, dim, max_mem_cnt, resp_mem_idx)

    new_shape = list(x.shape)
    new_shape[dim] = max_mem_cnt
    group_prob = probs.new_zeros((*new_shape,))
    group_prob.scatter_(dim, torch.where(is_mem, mem_group_idx - 1, max_mem_cnt - 1), probs)
    probs = probs.mul(
        torch.where(
            full_access_mask,
            last_section_mask.to(probs.dtype),
            torch.gather(group_prob, dim, resp_mem_idx),
        )
    )

    return probs
