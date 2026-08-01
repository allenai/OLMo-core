from dataclasses import dataclass

import torch

from olmo_core.config import Config
from olmo_core.distributed.nn import all_to_all, all_to_all_single
from olmo_core.distributed.utils import get_world_size


@dataclass
class ContextParallelConfig(Config):
    """
    Configuration class for context parallelism (CP).
    """

    degree: int
    """
    The CP degree.
    """


def all_to_all_single_cp2hp(
    input_: torch.Tensor, cp_group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    """
    Transform a tensor from context-parallel to head-parallel partitioning via AlltoAll.

    Ref: https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/ssm/mamba_context_parallel.py#L287

    :param input_: The input tensor with shape ``[B, T/CP, H, D]`` or ``[B, T/CP, H]``, partitioned
        along the sequence dimension across context parallel ranks. 3D inputs are treated as having
        D=1 (i.e., ``[B, T/CP, H, 1]``).
    :param cp_group: The process group for context parallel communication.
    :returns: The output tensor with shape ``[B, T, H/CP, D]`` or ``[B, T, H/CP]`` (matching input
        dimensionality), partitioned along the head dimension.
    """
    assert input_.dim() in (
        3,
        4,
    ), "all_to_all_single_cp2hp expects 3-d input shape [B, T/CP, H] or 4-d input shape [B, T/CP, H, D]."

    input_was_3d = input_.dim() == 3
    if input_was_3d:
        input_ = input_.unsqueeze(-1)  # [B, T/CP, H] -> [B, T/CP, H, 1]

    world_size = get_world_size(cp_group)

    B, t_local, h_in, d_in = input_.shape
    h_out = h_in // world_size

    # [B, T/CP, H, D] -> [B, T/CP, CP, H/CP, D] -> [CP, B, T/CP, H/CP, D]
    input_split = input_.view(B, t_local, world_size, h_out, d_in).permute(2, 0, 1, 3, 4)
    input_split = input_split.flatten(0, 3)

    exchanged = all_to_all_single(cp_group, input_split)

    # [CP, B, T/CP, H/CP, D] -> [B, CP, T/CP, H/CP, D] -> [B, T, H/CP, D]
    exchanged = exchanged.view(world_size, B, t_local, h_out, d_in).permute(1, 0, 2, 3, 4)
    # '.contiguous()' is not redundant: when the merged dims include one of size 1 (e.g. h_out == 1,
    # which happens as soon as the CP degree equals the head count) 'reshape' can satisfy the request
    # as a view and leaves the result non-contiguous, which then breaks a downstream '.view()'. When
    # the merge genuinely requires a copy, 'reshape' has already materialized it and this is a no-op.
    exchanged = exchanged.reshape(B, t_local * world_size, h_out, d_in).contiguous()

    if input_was_3d:
        exchanged = exchanged.squeeze(-1)  # [B, T, H/CP, 1] -> [B, T, H/CP]

    return exchanged


def all_to_all_cp2hp(
    inputs: list[torch.Tensor], cp_group: torch.distributed.ProcessGroup
) -> list[torch.Tensor]:
    """
    Transform multiple tensors from context-parallel to head-parallel partitioning via a single AlltoAll.

    This batches the communication into a single collective operation.

    :param inputs: List of input tensors, each with shape ``[B, T/CP, H, D]`` or ``[B, T/CP, H]``,
        partitioned along the sequence dimension across context parallel ranks. 3D inputs are
        treated as having D=1 (i.e., ``[B, T/CP, H, 1]``).
    :param cp_group: The process group for context parallel communication.
    :returns: List of output tensors, each with shape ``[B, T, H/CP, D]`` or ``[B, T, H/CP]``
        (matching input dimensionality), partitioned along the head dimension.
    """
    if not inputs:
        return []

    assert inputs[0].dim() in (
        3,
        4,
    ), "all_to_all_cp2hp expects 3-d input shape [B, T/CP, H] or 4-d input shape [B, T/CP, H, D]."
    inputs_were_3d = inputs[0].dim() == 3
    if inputs_were_3d:
        inputs = [input_.unsqueeze(-1) for input_ in inputs]  # [B, T/CP, H] -> [B, T/CP, H, 1]

    world_size = get_world_size(cp_group)

    # Validate and prepare all inputs: split each tensor into CP chunks
    prepared = []
    shapes = []
    for input_ in inputs:
        B, t_local, h_in, d_in = input_.shape
        h_out = h_in // world_size
        shapes.append((B, t_local, h_out, d_in))

        # [B, T/CP, H, D] -> [B, T/CP, CP, H/CP, D] -> [CP, B, T/CP, H/CP, D]
        input_split = input_.view(B, t_local, world_size, h_out, d_in).permute(2, 0, 1, 3, 4)
        prepared.append(input_split)

    # Build input list for all_to_all: concatenate chunks for each rank across all tensors
    # input_list[r] contains all chunks destined for rank r, concatenated
    chunk_size = prepared[0][0].numel()
    input_list = [torch.cat([p[r].flatten() for p in prepared], dim=0) for r in range(world_size)]

    output_list = all_to_all(cp_group, input_list)

    # Split received data back into individual tensors and reshape
    outputs = []
    for i, (B, t_local, h_out, d_in) in enumerate(shapes):
        chunks = [
            output_list[r][i * chunk_size : (i + 1) * chunk_size].view(B, t_local, h_out, d_in)
            for r in range(world_size)
        ]
        out = torch.stack(chunks, dim=0).permute(1, 0, 2, 3, 4)
        # See the note in 'all_to_all_single_cp2hp' on why '.contiguous()' is required here.
        out = out.reshape(B, t_local * world_size, h_out, d_in).contiguous()
        outputs.append(out)

    if inputs_were_3d:
        outputs = [out.squeeze(-1) for out in outputs]  # [B, T, H/CP, 1] -> [B, T, H/CP]

    return outputs


def _repeat_kv_heads(input_: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Expand the head dimension of a ``[B, T, H_kv, D]`` tensor by ``n_rep`` in GQA order, i.e.
    ``[kv_0] * n_rep + [kv_1] * n_rep + ...``.

    Implemented with ``expand`` + ``reshape`` (rather than a materializing copy) so that autograd
    sums the gradients back across the replicas on the backward pass. That is what makes the
    replication in :func:`all_to_all_qkv_cp2hp` correct without any hand-written gradient reduction.

    :param input_: The tensor to expand, with shape ``[B, T, H_kv, D]``.
    :param n_rep: The number of times to repeat each KV head.

    :returns: A tensor with shape ``[B, T, H_kv * n_rep, D]``.
    """
    if n_rep == 1:
        return input_
    B, T, h_kv, d = input_.shape
    return input_.unsqueeze(3).expand(B, T, h_kv, n_rep, d).reshape(B, T, h_kv * n_rep, d)


def all_to_all_qkv_cp2hp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cp_group: torch.distributed.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Transform ``q``, ``k`` and ``v`` from context-parallel to head-parallel partitioning for Ulysses
    attention, replicating KV heads when the CP degree does not divide ``n_kv_heads``.

    Ulysses gathers the *full* sequence on every rank, so the head split is purely a work
    partitioning choice: nothing about the attention math requires the CP degree to divide
    ``n_kv_heads``. When it doesn't (typically because the CP degree exceeds ``n_kv_heads`` on a GQA
    model), each KV head is first replicated ``n_heads // n_kv_heads`` times so that the split is
    well defined. Rank ``r`` then receives query heads ``[r * H/CP, (r+1) * H/CP)`` alongside exactly
    the replicated KV heads those queries attend to, since :func:`all_to_all_cp2hp` splits heads
    contiguously and :func:`_repeat_kv_heads` lays replicas out in matching GQA order.

    When the CP degree already divides ``n_kv_heads`` this is exactly the plain two-collective path,
    with no extra communication.

    :param q: Query tensor with shape ``[B, T/CP, n_heads, D]``.
    :param k: Key tensor with shape ``[B, T/CP, n_kv_heads, D]``.
    :param v: Value tensor with shape ``[B, T/CP, n_kv_heads, D]``.
    :param cp_group: The process group for context parallel communication.

    :returns: ``(q, k, v)`` with shapes ``[B, T, n_heads/CP, D]`` and, for ``k``/``v``, either
        ``[B, T, n_kv_heads/CP, D]`` (no replication) or ``[B, T, n_heads/CP, D]`` (replicated).

    :raises RuntimeError: If the CP degree doesn't divide ``n_heads``, or if ``n_kv_heads`` doesn't
        divide ``n_heads``.
    """
    world_size = get_world_size(cp_group)
    n_heads, n_kv_heads = q.shape[2], k.shape[2]

    if n_heads % world_size != 0:
        raise RuntimeError(
            f"Ulysses CP degree ({world_size}) must divide the number of query heads ({n_heads})"
        )

    # Only replicate when the plain split isn't well defined; otherwise keep the cheaper path so
    # existing configurations are unchanged, both numerically and in communication volume.
    if n_kv_heads % world_size != 0:
        if n_heads % n_kv_heads != 0:
            raise RuntimeError(
                f"Number of KV heads ({n_kv_heads}) must divide the number of query heads "
                f"({n_heads}) to replicate KV heads across {world_size} context parallel ranks"
            )
        n_rep = n_heads // n_kv_heads
        k = _repeat_kv_heads(k, n_rep)
        v = _repeat_kv_heads(v, n_rep)

    q = all_to_all_single_cp2hp(q, cp_group)
    k, v = all_to_all_cp2hp([k, v], cp_group)
    return q, k, v


def all_to_all_single_hp2cp(
    input_: torch.Tensor, cp_group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    """
    Transform a tensor from head-parallel to context-parallel partitioning via AlltoAll.

    Ref: https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/ssm/mamba_context_parallel.py#L324

    :param input_: The input tensor with shape ``[B, T, H/CP, D]`` or ``[B, T, H/CP]``, containing
        full sequence but partitioned along the head dimension. 3D inputs are treated as having
        D=1 (i.e., ``[B, T, H/CP, 1]``).
    :param cp_group: The process group for context parallel communication.
    :returns: The output tensor with shape ``[B, T/CP, H, D]`` or ``[B, T/CP, H]`` (matching input
        dimensionality), partitioned along the sequence dimension but with full head dimension.
    """
    assert input_.dim() in (
        3,
        4,
    ), "all_to_all_single_hp2cp expects 3-d input shape [B, T, H/CP] or 4-d input shape [B, T, H/CP, D]."

    input_was_3d = input_.dim() == 3
    if input_was_3d:
        input_ = input_.unsqueeze(-1)  # [B, T, H/CP] -> [B, T, H/CP, 1]

    world_size = get_world_size(cp_group)

    B, t_full, h_in, d_in = input_.shape
    t_out = t_full // world_size

    # [B, T, H/CP, D] -> [B, CP, T/CP, H/CP, D] -> [CP, B, T/CP, H/CP, D]
    input_split = input_.view(B, world_size, t_out, h_in, d_in).permute(1, 0, 2, 3, 4)
    input_split = input_split.flatten(0, 3)

    exchanged = all_to_all_single(cp_group, input_split)

    # [CP, B, T/CP, H/CP, D] -> [B, T/CP, CP, H/CP, D] -> [B, T/CP, H, D]
    exchanged = exchanged.view(world_size, B, t_out, h_in, d_in).permute(1, 2, 0, 3, 4)
    # See the note in 'all_to_all_single_cp2hp' on why '.contiguous()' is required here.
    exchanged = exchanged.reshape(B, t_out, h_in * world_size, d_in).contiguous()

    if input_was_3d:
        exchanged = exchanged.squeeze(-1)  # [B, T/CP, H, 1] -> [B, T/CP, H]

    return exchanged


def all_to_all_hp2cp(
    inputs: list[torch.Tensor], cp_group: torch.distributed.ProcessGroup
) -> list[torch.Tensor]:
    """
    Transform multiple tensors from head-parallel to context-parallel partitioning via a single AlltoAll.

    This is more efficient than calling :func:`all_to_all_hp2cp` multiple times as it batches
    the communication into a single collective operation.

    :param inputs: List of input tensors, each with shape ``[B, T, H/CP, D]`` or ``[B, T, H/CP]``,
        containing full sequence but partitioned along the head dimension. 3D inputs are treated
        as having D=1 (i.e., ``[B, T, H/CP, 1]``).
    :param cp_group: The process group for context parallel communication.
    :returns: List of output tensors, each with shape ``[B, T/CP, H, D]`` or ``[B, T/CP, H]``
        (matching input dimensionality), partitioned along the sequence dimension but with full
        head dimension.
    """
    if not inputs:
        return []

    assert inputs[0].dim() in (
        3,
        4,
    ), "all_to_all_hp2cp expects 3-d input shape [B, T, H/CP] or 4-d input shape [B, T, H/CP, D]."
    inputs_were_3d = inputs[0].dim() == 3
    if inputs_were_3d:
        inputs = [input_.unsqueeze(-1) for input_ in inputs]  # [B, T, H/CP] -> [B, T, H/CP, 1]

    world_size = get_world_size(cp_group)

    # Validate and prepare all inputs: split each tensor into CP chunks
    prepared = []
    shapes = []
    for input_ in inputs:
        B, t_full, h_in, d_in = input_.shape
        t_out = t_full // world_size
        shapes.append((B, t_out, h_in, d_in))

        # [B, T, H/CP, D] -> [B, CP, T/CP, H/CP, D] -> [CP, B, T/CP, H/CP, D]
        input_split = input_.view(B, world_size, t_out, h_in, d_in).permute(1, 0, 2, 3, 4)
        prepared.append(input_split)

    # Build input list for all_to_all: concatenate chunks for each rank across all tensors
    # input_list[r] contains all chunks destined for rank r, concatenated
    chunk_size = prepared[0][0].numel()
    input_list = [torch.cat([p[r].flatten() for p in prepared], dim=0) for r in range(world_size)]

    output_list = all_to_all(cp_group, input_list)

    # Split received data back into individual tensors and reshape
    outputs = []
    for i, (B, t_out, h_in, d_in) in enumerate(shapes):
        chunks = [
            output_list[r][i * chunk_size : (i + 1) * chunk_size].view(B, t_out, h_in, d_in)
            for r in range(world_size)
        ]
        out = torch.stack(chunks, dim=0).permute(1, 2, 0, 3, 4)
        # See the note in 'all_to_all_single_cp2hp' on why '.contiguous()' is required here.
        out = out.reshape(B, t_out, h_in * world_size, d_in).contiguous()
        outputs.append(out)

    if inputs_were_3d:
        outputs = [out.squeeze(-1) for out in outputs]  # [B, T/CP, H, 1] -> [B, T/CP, H]

    return outputs


def all_to_all_single_cp2hp_qkvpacked(
    input_: torch.Tensor, cp_group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    """
    Transform a packed QKV tensor from context-parallel to head-parallel partitioning via AlltoAll.

    :param input_: The input tensor with shape ``[B, T/CP, 3, H, D]``, partitioned along
        the sequence dimension across context parallel ranks.
    :param cp_group: The process group for context parallel communication.
    :returns: The output tensor with shape ``[B, T, 3, H/CP, D]``, partitioned along
        the head dimension.

    :raises RuntimeError: If the CP degree doesn't divide the number of heads. Packed QKV implies
        ``n_heads == n_kv_heads``, so unlike :func:`all_to_all_qkv_cp2hp` there is no KV replication
        that could rescue this — the CP degree is simply too large for the model.
    """
    assert (
        input_.dim() == 5
    ), "all_to_all_single_cp2hp_qkvpacked expects 5-d input shape [B, T/CP, 3, H, D]."
    world_size = get_world_size(cp_group)

    B, t_local, three, h_in, d_in = input_.shape
    assert three == 3
    if h_in % world_size != 0:
        raise RuntimeError(
            f"Ulysses CP degree ({world_size}) must divide the number of heads ({h_in}) "
            "when using packed QKV"
        )
    h_out = h_in // world_size

    # [B, T/CP, 3, H, D] -> [B, T/CP, 3, CP, H/CP, D] -> [CP, B, T/CP, 3, H/CP, D]
    input_split = input_.view(B, t_local, three, world_size, h_out, d_in).permute(3, 0, 1, 2, 4, 5)
    input_split = input_split.flatten(0, 4)

    exchanged = all_to_all_single(cp_group, input_split)

    # [CP, B, T/CP, 3, H/CP, D] -> [B, CP, T/CP, 3, H/CP, D] -> [B, T, 3, H/CP, D]
    exchanged = exchanged.view(world_size, B, t_local, three, h_out, d_in).permute(1, 0, 2, 3, 4, 5)
    # See the note in 'all_to_all_single_cp2hp' on why '.contiguous()' is required here.
    exchanged = exchanged.reshape(B, t_local * world_size, three, h_out, d_in).contiguous()
    return exchanged


def all_to_all_single_hp2cp_qkvpacked(
    input_: torch.Tensor, cp_group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    """
    Transform a packed QKV tensor from head-parallel to context-parallel partitioning via AlltoAll.

    :param input_: The input tensor with shape ``[B, T, 3, H/CP, D]``, containing full sequence
        but partitioned along the head dimension.
    :param cp_group: The process group for context parallel communication.
    :returns: The output tensor with shape ``[B, T/CP, 3, H, D]``, partitioned along
        the sequence dimension but with full head dimension.
    """
    assert (
        input_.dim() == 5
    ), "all_to_all_single_hp2cp_qkvpacked expects 5-d input shape [B, T, 3, H/CP, D]."
    world_size = get_world_size(cp_group)

    B, t_full, three, h_in, d_in = input_.shape
    assert three == 3
    t_out = t_full // world_size

    # [B, T, 3, H/CP, D] -> [B, CP, T/CP, 3, H/CP, D] -> [CP, B, T/CP, 3, H/CP, D]
    input_split = input_.view(B, world_size, t_out, three, h_in, d_in).permute(1, 0, 2, 3, 4, 5)
    input_split = input_split.flatten(0, 4)

    exchanged = all_to_all_single(cp_group, input_split)

    # [CP, B, T/CP, 3, H/CP, D] -> [B, T/CP, CP, 3, H/CP, D] -> [B, T/CP, 3, H, D]
    exchanged = exchanged.view(world_size, B, t_out, three, h_in, d_in).permute(1, 2, 0, 3, 4, 5)
    # See the note in 'all_to_all_single_cp2hp' on why '.contiguous()' is required here.
    exchanged = exchanged.reshape(B, t_out, three, h_in * world_size, d_in).contiguous()
    return exchanged
