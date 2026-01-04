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
    input_: torch.Tensor, cp_group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    """
    Transform a tensor from context-parallel to head-parallel partitioning via AlltoAll.

    This function redistributes tensor data across a context parallel process group,
    converting from sequence-partitioned to hidden-dimension-partitioned layout.
    This is typically used in attention mechanisms to gather the full sequence
    while distributing across attention heads.

    Ref: https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/ssm/mamba_context_parallel.py#L287

    Input shape: ``[T/CP, B, H]`` → Output shape: ``[T, B, H/CP]``

    :param input_: The input tensor with shape ``[T/CP, B, H]``, partitioned along
        the sequence dimension across context parallel ranks.
    :param cp_group: The process group for context parallel communication.

    :returns: The output tensor with shape ``[T, B, H/CP]``, now containing the full
        sequence but partitioned along the hidden dimension.
    """
    assert input_.dim() == 3, "all_to_all_cp2hp assumes 3-d input shape."
    t_in, b_in, h_in = input_.shape
    input_ = input_.reshape(-1, h_in)  # [t/CP*b, h]

    world_size = cp_group.size()
    h_out = h_in // world_size
    split_tensors = torch.split(input_, split_size_or_sections=h_out, dim=1)  # [t/CP*b, h/CP] * CP

    # all-to-all on a single tensor.
    concat_tensor = torch.cat(split_tensors, dim=0)  # [t/CP*b*CP, h/CP] = [t*b, h/CP]
    output = all_to_all(cp_group, concat_tensor)  # [t*b, h/CP]

    # Recover the t and b dimensions
    output = output.reshape(t_in * world_size, b_in, h_out)  # [t, b, h/CP]
    return output


def all_to_all_hp2cp(
    input_: torch.Tensor, cp_group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    """
    Transform a tensor from head-parallel to context-parallel partitioning via AlltoAll.

    This function redistributes tensor data across a context parallel process group,
    converting from hidden-dimension-partitioned to sequence-partitioned layout.
    This is typically used in attention mechanisms to distribute the sequence
    while gathering across attention heads.

    Ref: https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/ssm/mamba_context_parallel.py#L324

    Input shape: ``[T, B, H/CP]`` → Output shape: ``[T/CP, B, H]``

    :param input_: The input tensor with shape ``[T, B, H/CP]``, partitioned along
        the hidden dimension across context parallel ranks.
    :param cp_group: The process group for context parallel communication.

    :returns: The output tensor with shape ``[T/CP, B, H]``, now containing the full
        hidden dimension but partitioned along the sequence.
    """
    assert input_.dim() == 3, "all_to_all_hp2cp assumes 3-d input shape."
    t_in, b_in, h_in = input_.shape
    input_ = input_.reshape(-1, h_in)  # [t*b, h/CP]

    world_size = cp_group.size()
    t_out = t_in // world_size
    split_tensors = torch.split(
        input_, split_size_or_sections=t_out * b_in, dim=0
    )  # [t/CP*b, h/CP] * CP

    # all-to-all on a single tensor.
    concat_tensor = torch.cat(split_tensors, dim=1)  # [t/CP*b, h/CP*CP] = [t/CP*b, h]
    output = all_to_all(cp_group, concat_tensor)  # [t/CP*b, h]

    # Recover the t and b dimensions
    output = output.reshape(t_out, b_in, h_in * world_size)  # [t/CP, b, h]
    return output
