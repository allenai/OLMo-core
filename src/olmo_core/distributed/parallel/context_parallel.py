from dataclasses import dataclass

import torch

from olmo_core.config import Config
from olmo_core.distributed.autograd import all_to_all


@dataclass
class ContextParallelConfig(Config):
    """
    Configuration class for context parallelism (CP).
    """

    degree: int
    """
    The CP degree.
    """


def all_to_all_cp2hp(
    input_: torch.Tensor, cp_group: torch.distributed.ProcessGroup, scatter_dim: int = 2
) -> torch.Tensor:
    """
    Transform a tensor from context-parallel to head-parallel partitioning via AlltoAll.

    This function redistributes tensor data across a context parallel process group,
    converting from sequence-partitioned to hidden-dimension-partitioned layout.
    This is typically used in attention mechanisms to gather the full sequence
    while distributing across attention heads.

    Ref: https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/ssm/mamba_context_parallel.py#L287

    When ``scatter_dim=2``: Input shape ``[T/CP, B, H]`` → Output shape ``[T, B, H/CP]``
    When ``scatter_dim=1``: Input shape ``[T/CP, H, D]`` → Output shape ``[T, H/CP, D]``

    :param input_: The input tensor with shape ``[T/CP, ...]``, partitioned along
        the sequence dimension across context parallel ranks.
    :param cp_group: The process group for context parallel communication.
    :param scatter_dim: The dimension to scatter across CP ranks. Default is 2 (last dim).
        Use 1 for attention tensors with shape ``[T, n_heads, head_dim]``.

    :returns: The output tensor with full sequence but partitioned along ``scatter_dim``.
    """
    assert input_.dim() == 3, "all_to_all_cp2hp assumes 3-d input shape."
    world_size = cp_group.size()

    if scatter_dim == 2:
        # Scatter along last dimension
        t_in, b_in, h_in = input_.shape
        input_ = input_.reshape(-1, h_in)  # [t/CP*b, h]

        h_out = h_in // world_size
        split_tensors = torch.split(
            input_, split_size_or_sections=h_out, dim=1
        )  # [t/CP*b, h/CP] * CP

        # all-to-all on a single tensor.
        concat_tensor = torch.cat(split_tensors, dim=0)  # [t/CP*b*CP, h/CP] = [t*b, h/CP]
        output = all_to_all(cp_group, concat_tensor)  # [t*b, h/CP]

        # Recover the t and b dimensions
        output = output.reshape(t_in * world_size, b_in, h_out)  # [t, b, h/CP]
    elif scatter_dim == 1:
        # Scatter along middle dimension (for attention: [T, n_heads, head_dim])
        t_in, h_in, d_in = input_.shape
        input_ = input_.reshape(-1, d_in)  # [t/CP*h, d]

        h_out = h_in // world_size
        # Split along dim 0, taking h_out rows at a time for each of t_in positions
        # Reshape to [t/CP, h, d] -> split h -> [t/CP, h/CP, d] * CP
        input_3d = input_.reshape(t_in, h_in, d_in)
        split_tensors = torch.split(
            input_3d, split_size_or_sections=h_out, dim=1
        )  # [t/CP, h/CP, d] * CP

        # Concatenate along sequence dimension
        concat_tensor = torch.cat(split_tensors, dim=0)  # [t/CP*CP, h/CP, d] = [t, h/CP, d]
        concat_tensor = concat_tensor.reshape(-1, d_in)  # [t*h/CP, d]
        output = all_to_all(cp_group, concat_tensor)  # [t*h/CP, d]

        # Recover dimensions
        output = output.reshape(t_in * world_size, h_out, d_in)  # [t, h/CP, d]
    else:
        raise ValueError(f"scatter_dim must be 1 or 2, got {scatter_dim}")

    return output


def all_to_all_hp2cp(
    input_: torch.Tensor, cp_group: torch.distributed.ProcessGroup, gather_dim: int = 2
) -> torch.Tensor:
    """
    Transform a tensor from head-parallel to context-parallel partitioning via AlltoAll.

    This function redistributes tensor data across a context parallel process group,
    converting from hidden-dimension-partitioned to sequence-partitioned layout.
    This is typically used in attention mechanisms to distribute the sequence
    while gathering across attention heads.

    Ref: https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/ssm/mamba_context_parallel.py#L324

    When ``gather_dim=2``: Input shape ``[T, B, H/CP]`` → Output shape ``[T/CP, B, H]``
    When ``gather_dim=1``: Input shape ``[T, H/CP, D]`` → Output shape ``[T/CP, H, D]``

    :param input_: The input tensor with shape ``[T, ...]``, containing full sequence
        but partitioned along ``gather_dim``.
    :param cp_group: The process group for context parallel communication.
    :param gather_dim: The dimension to gather across CP ranks. Default is 2 (last dim).
        Use 1 for attention tensors with shape ``[T, n_heads, head_dim]``.

    :returns: The output tensor with shape ``[T/CP, ...]``, partitioned along
        the sequence dimension but with full ``gather_dim``.
    """
    assert input_.dim() == 3, "all_to_all_hp2cp assumes 3-d input shape."
    world_size = cp_group.size()

    if gather_dim == 2:
        # Original behavior: gather along last dimension
        t_in, b_in, h_in = input_.shape
        input_ = input_.reshape(-1, h_in)  # [t*b, h/CP]

        t_out = t_in // world_size
        split_tensors = torch.split(
            input_, split_size_or_sections=t_out * b_in, dim=0
        )  # [t/CP*b, h/CP] * CP

        # all-to-all on a single tensor.
        concat_tensor = torch.cat(split_tensors, dim=1)  # [t/CP*b, h/CP*CP] = [t/CP*b, h]
        output = all_to_all(cp_group, concat_tensor)  # [t/CP*b, h]

        # Recover the t and b dimensions
        output = output.reshape(t_out, b_in, h_in * world_size)  # [t/CP, b, h]
    elif gather_dim == 1:
        # Gather along middle dimension (for attention: [T, n_heads, head_dim])
        t_in, h_in, d_in = input_.shape
        t_out = t_in // world_size

        # Split along sequence dimension
        input_3d = input_.reshape(t_in, h_in, d_in)
        split_tensors = torch.split(
            input_3d, split_size_or_sections=t_out, dim=0
        )  # [t/CP, h/CP, d] * CP

        # Concatenate along heads dimension
        concat_tensor = torch.cat(split_tensors, dim=1)  # [t/CP, h/CP*CP, d] = [t/CP, h, d]
        concat_tensor = concat_tensor.reshape(-1, d_in)  # [t/CP*h, d]
        output = all_to_all(cp_group, concat_tensor)  # [t/CP*h, d]

        # Recover dimensions
        output = output.reshape(t_out, h_in * world_size, d_in)  # [t/CP, h, d]
    else:
        raise ValueError(f"gather_dim must be 1 or 2, got {gather_dim}")

    return output
