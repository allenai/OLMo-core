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

import torch

__all__ = [
    "repeat_kv",
    "landmark_grouped_softmax",
    "LandmarkGroupedSoftmaxFunction",
    "build_block_doc_id",
    "build_local_packed_position_ids",
]


def build_block_doc_id(
    cu_doc_lens: torch.Tensor,
    batch_size: int,
    seq_len: int,
    block_size: int,
) -> torch.Tensor:
    """
    Build a per-landmark-block document id for sequence packing, used by the fused landmark kernels.

    ``cu_doc_lens`` follows the flattened-over-batch convention of the flash-attention backend (see
    :func:`olmo_core.data.utils.get_cumulative_document_lengths`): the cumulative sum of every
    document length across the whole ``(batch_size, seq_len)`` micro-batch, i.e.
    ``[0, ..., batch_size * seq_len]``. Because document boundaries are required to be block-aligned,
    every landmark block (``block_size`` consecutive tokens) lies in exactly one document, so a single
    id per block suffices.

    :param cu_doc_lens: 1D cumulative document lengths over the flattened batch.
    :param batch_size: Number of sequences in the micro-batch.
    :param seq_len: Per-sequence length ``T`` (a multiple of ``block_size``).
    :param block_size: Landmark block size (``mem_freq + 1``).

    :returns: An int32 tensor of shape ``(batch_size, seq_len // block_size)`` giving each block's
        document id (monotonic within each row; absolute values are irrelevant, only equality is used).

    :raises ValueError: If a boundary is not block-aligned or ``cu_doc_lens`` does not total
        ``batch_size * seq_len`` tokens.
    """
    boundaries = cu_doc_lens.to(dtype=torch.long)
    block_size_t = block_size
    if bool((boundaries % block_size_t != 0).any().item()):
        raise ValueError(
            f"Landmark packing requires every document boundary to be a multiple of the landmark "
            f"block size ({block_size}), but got cu_doc_lens={cu_doc_lens.tolist()}. This almost "
            f"always means the document lengths came from the wrong source: EOS-derived "
            f"'generate_doc_lengths' or a generic token packer (PackingInstanceSource / "
            f"ConcatAndChunk) do NOT produce block-aligned landmark documents. Build packed landmark "
            f"SFT data with LandmarkPackingInstanceSource (it inserts landmarks per document and "
            f"emits block-aligned doc_lens) and set generate_doc_lengths=False."
        )
    total = batch_size * seq_len
    if int(boundaries[-1].item()) != total:
        raise ValueError(
            f"cu_doc_lens must total batch_size * seq_len = {total} tokens (flattened over the "
            f"batch), but its last entry is {int(boundaries[-1].item())}."
        )
    n_blocks = seq_len // block_size
    interior = boundaries[(boundaries > 0) & (boundaries < total)]
    # Flat position of each block's first token: z * seq_len + b * block_size.
    z = torch.arange(batch_size, device=cu_doc_lens.device)
    b = torch.arange(n_blocks, device=cu_doc_lens.device)
    flat_first = z[:, None] * seq_len + b[None, :] * block_size  # (batch_size, n_blocks)
    doc_id = torch.searchsorted(interior, flat_first, right=True)
    return doc_id.to(torch.int32)


def build_local_packed_position_ids(
    cu_doc_lens: torch.Tensor,
    batch_size: int,
    seq_len_local: int,
    cp_rank: int,
    cp_world_size: int,
) -> torch.Tensor:
    """
    Build per-document RoPE positions for a Ulysses context-parallel shard of a packed sequence.

    Under Ulysses CP the sequence is sharded *contiguously*: rank ``r`` holds the global sequence
    positions ``[r * seq_len_local, (r + 1) * seq_len_local)`` of every batch row, while
    ``cu_doc_lens`` still describes the **full** (unsharded) ``(batch_size, seq_len)`` micro-batch --
    the Ulysses load balancer passes the document boundaries through unchanged
    (see :meth:`~olmo_core.nn.attention.ring.UlyssesLoadBalancer.batch_shard_by_document`). RoPE is
    applied on the local shard *before* the Ulysses all-to-all gathers the full sequence, so each
    local token needs its position *within its document*. For a document that straddles a rank
    boundary that position continues across the boundary (it does **not** reset to 0 at the start of
    the shard), which is exactly what this function computes.

    This mirrors the per-document position reset of
    :meth:`~olmo_core.nn.rope.RotaryEmbedding.forward` (the ``cu_doc_lens`` branch), but offset to
    the rank's slice of the global sequence so straddling documents stay continuous.

    :param cu_doc_lens: 1D cumulative document lengths over the flattened full ``(batch_size,
        seq_len)`` micro-batch (flash-attention convention), i.e. ``[0, ..., batch_size * seq_len]``.
    :param batch_size: Number of sequences in the micro-batch.
    :param seq_len_local: Per-rank (sharded) sequence length ``seq_len // cp_world_size``.
    :param cp_rank: This rank's index within the CP process group.
    :param cp_world_size: The CP degree.

    :returns: An int64 tensor of shape ``(batch_size, seq_len_local)`` giving each local token's
        position within its (global) document.
    """
    seq_len = seq_len_local * cp_world_size
    device = cu_doc_lens.device
    boundaries = cu_doc_lens.to(dtype=torch.long)
    z = torch.arange(batch_size, device=device)[:, None]
    t = torch.arange(seq_len_local, device=device)[None, :]
    # Flattened-over-batch global position of local token (z, t): z * seq_len + (rank shard offset).
    global_flat = z * seq_len + cp_rank * seq_len_local + t  # (batch_size, seq_len_local)
    flat = global_flat.reshape(-1)
    doc_id = torch.bucketize(flat, boundaries[1:], right=True)
    pos = flat - boundaries[doc_id]
    return pos.view(batch_size, seq_len_local)


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
