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

    :param input_: The input tensor with shape ``[B, T/CP, H, D]``, partitioned along
        the sequence dimension across context parallel ranks.
    :param cp_group: The process group for context parallel communication.
    :returns: The output tensor with shape ``[B, T, H/CP, D]``, partitioned along
        the head dimension.
    """
    assert input_.dim() == 4, "all_to_all_cp2hp expects 4-d input shape [B, T/CP, H, D]."
    world_size = get_world_size(cp_group)

    B, t_local, h_in, d_in = input_.shape
    h_out = h_in // world_size

    # [B, T/CP, H, D] -> [B, T/CP, CP, H/CP, D] -> [CP, B, T/CP, H/CP, D]
    input_split = input_.view(B, t_local, world_size, h_out, d_in).permute(2, 0, 1, 3, 4)
    input_split = input_split.flatten(0, 3)

    exchanged = all_to_all_single(cp_group, input_split)

    # [CP, B, T/CP, H/CP, D] -> [B, CP, T/CP, H/CP, D] -> [B, T, H/CP, D]
    exchanged = exchanged.view(world_size, B, t_local, h_out, d_in).permute(1, 0, 2, 3, 4)
    exchanged = exchanged.reshape(B, t_local * world_size, h_out, d_in)
    return exchanged


def all_to_all_cp2hp(
    inputs: list[torch.Tensor], cp_group: torch.distributed.ProcessGroup
) -> list[torch.Tensor]:
    """
    Transform multiple tensors from context-parallel to head-parallel partitioning via a single AlltoAll.

    This batches the communication into a single collective operation.

    :param inputs: List of input tensors, each with shape ``[B, T/CP, H, D]``, partitioned along
        the sequence dimension across context parallel ranks.
    :param cp_group: The process group for context parallel communication.
    :returns: List of output tensors, each with shape ``[B, T, H/CP, D]``, partitioned along
        the head dimension.
    """
    if not inputs:
        return []

    world_size = get_world_size(cp_group)

    # Validate and prepare all inputs: split each tensor into CP chunks
    # prepared[i] has shape [CP, B, T/CP, H/CP, D] for input i
    prepared = []
    shapes = []
    for input_ in inputs:
        assert input_.dim() == 4, "all_to_all_cp2hp expects 4-d input shape [B, T/CP, H, D]."
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

    # Single all-to-all exchange
    output_list = all_to_all(cp_group, input_list)

    # Split received data back into individual tensors and reshape
    outputs = []
    for i, (B, t_local, h_out, d_in) in enumerate(shapes):
        # Extract this tensor's data from each rank and stack
        chunks = [
            output_list[r][i * chunk_size : (i + 1) * chunk_size].view(B, t_local, h_out, d_in)
            for r in range(world_size)
        ]
        # Stack along new dim then merge: [CP, B, T/CP, H/CP, D] -> [B, T, H/CP, D]
        out = torch.stack(chunks, dim=0).permute(1, 0, 2, 3, 4)
        outputs.append(out.reshape(B, t_local * world_size, h_out, d_in))

    return outputs


def all_to_all_single_hp2cp(
    input_: torch.Tensor, cp_group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    """
    Transform a tensor from head-parallel to context-parallel partitioning via AlltoAll.

    Ref: https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/ssm/mamba_context_parallel.py#L324

    :param input_: The input tensor with shape ``[B, T, H/CP, D]``, containing full sequence
        but partitioned along the head dimension.
    :param cp_group: The process group for context parallel communication.
    :returns: The output tensor with shape ``[B, T/CP, H, D]``, partitioned along
        the sequence dimension but with full head dimension.
    """
    assert input_.dim() == 4, "all_to_all_single_hp2cp expects 4-d input shape [B, T, H/CP, D]."
    world_size = get_world_size(cp_group)

    B, t_full, h_in, d_in = input_.shape
    t_out = t_full // world_size

    # [B, T, H/CP, D] -> [B, CP, T/CP, H/CP, D] -> [CP, B, T/CP, H/CP, D]
    input_split = input_.view(B, world_size, t_out, h_in, d_in).permute(1, 0, 2, 3, 4)
    input_split = input_split.flatten(0, 3)

    exchanged = all_to_all_single(cp_group, input_split)

    # [CP, B, T/CP, H/CP, D] -> [B, T/CP, CP, H/CP, D] -> [B, T/CP, H, D]
    exchanged = exchanged.view(world_size, B, t_out, h_in, d_in).permute(1, 2, 0, 3, 4)
    exchanged = exchanged.reshape(B, t_out, h_in * world_size, d_in)
    return exchanged


def all_to_all_hp2cp(
    inputs: list[torch.Tensor], cp_group: torch.distributed.ProcessGroup
) -> list[torch.Tensor]:
    """
    Transform multiple tensors from head-parallel to context-parallel partitioning via a single AlltoAll.

    This is more efficient than calling :func:`all_to_all_hp2cp` multiple times as it batches
    the communication into a single collective operation.

    :param inputs: List of input tensors, each with shape ``[B, T, H/CP, D]``, containing full sequence
        but partitioned along the head dimension.
    :param cp_group: The process group for context parallel communication.
    :returns: List of output tensors, each with shape ``[B, T/CP, H, D]``, partitioned along
        the sequence dimension but with full head dimension.
    """
    if not inputs:
        return []

    world_size = get_world_size(cp_group)

    # Validate and prepare all inputs: split each tensor into CP chunks
    # prepared[i] has shape [CP, B, T/CP, H/CP, D] for input i
    prepared = []
    shapes = []
    for input_ in inputs:
        assert input_.dim() == 4, "all_to_all_hp2cp expects 4-d input shape [B, T, H/CP, D]."
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

    # Single all-to-all exchange
    output_list = all_to_all(cp_group, input_list)

    # Split received data back into individual tensors and reshape
    outputs = []
    for i, (B, t_out, h_in, d_in) in enumerate(shapes):
        # Extract this tensor's data from each rank and stack
        chunks = [
            output_list[r][i * chunk_size : (i + 1) * chunk_size].view(B, t_out, h_in, d_in)
            for r in range(world_size)
        ]
        # Stack along new dim then permute: [CP, B, T/CP, H/CP, D] -> [B, T/CP, CP, H/CP, D] -> [B, T/CP, H, D]
        out = torch.stack(chunks, dim=0).permute(1, 2, 0, 3, 4)
        outputs.append(out.reshape(B, t_out, h_in * world_size, d_in))

    return outputs


def all_to_all_cp2hp_qkvpacked(
    input_: torch.Tensor, cp_group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    """
    Transform a packed QKV tensor from context-parallel to head-parallel partitioning via AlltoAll.

    :param input_: The input tensor with shape ``[B, T/CP, 3, H, D]``, partitioned along
        the sequence dimension across context parallel ranks.
    :param cp_group: The process group for context parallel communication.
    :returns: The output tensor with shape ``[B, T, 3, H/CP, D]``, partitioned along
        the head dimension.
    """
    assert (
        input_.dim() == 5
    ), "all_to_all_single_cp2hp_qkvpacked expects 5-d input shape [B, T/CP, 3, H, D]."
    world_size = get_world_size(cp_group)

    B, t_local, three, h_in, d_in = input_.shape
    assert three == 3
    h_out = h_in // world_size

    # [B, T/CP, 3, H, D] -> [B, T/CP, 3, CP, H/CP, D] -> [CP, B, T/CP, 3, H/CP, D]
    input_split = input_.view(B, t_local, three, world_size, h_out, d_in).permute(3, 0, 1, 2, 4, 5)
    input_split = input_split.flatten(0, 4)

    exchanged = all_to_all_single(cp_group, input_split)

    # [CP, B, T/CP, 3, H/CP, D] -> [B, CP, T/CP, 3, H/CP, D] -> [B, T, 3, H/CP, D]
    exchanged = exchanged.view(world_size, B, t_local, three, h_out, d_in).permute(1, 0, 2, 3, 4, 5)
    exchanged = exchanged.reshape(B, t_local * world_size, three, h_out, d_in)
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
    exchanged = exchanged.reshape(B, t_out, three, h_in * world_size, d_in)
    return exchanged
