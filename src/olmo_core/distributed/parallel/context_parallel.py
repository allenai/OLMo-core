from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from olmo_core.config import Config
from olmo_core.distributed.autograd import all_to_all
from olmo_core.distributed.utils import get_world_size

if TYPE_CHECKING:
    from olmo_core.nn.attention.ring import RingAttentionLoadBalancerType


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

    Ref: https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/ssm/mamba_context_parallel.py#L287


    :param input_: The input tensor with shape ``[BT/CP, H, D]``, partitioned along
        the sequence dimension across context parallel ranks.
    :param cp_group: The process group for context parallel communication.
    :returns: The output tensor with shape ``[BT, H/CP, D]``, partitioned along
        the sequence dimension but with full head dimension.
    """
    assert input_.dim() == 3, "all_to_all_cp2hp assumes 3-d input shape."
    world_size = get_world_size(cp_group)

    # Scatter along middle dimension (for attention: [BT, n_heads, head_dim])
    bt_in, h_in, d_in = input_.shape
    h_out = h_in // world_size

    # [bt/CP, CP, h/CP, d] -> [CP, bt/CP, h/CP, d] where dim0 indexes destination rank.
    input_split = input_.reshape(bt_in, world_size, h_out, d_in).permute(1, 0, 2, 3)
    flattened = input_split.flatten(0, 2)  # [CP*bt/CP*h/CP, d]
    exchanged = all_to_all(cp_group, flattened)  # [CP * bt/CP * h/CP, d]

    output = exchanged.reshape(bt_in * world_size, h_out, d_in)  # [bt, h/CP, d]

    return output


def all_to_all_hp2cp(
    input_: torch.Tensor, cp_group: torch.distributed.ProcessGroup
) -> torch.Tensor:
    """
    Transform a tensor from head-parallel to context-parallel partitioning via AlltoAll.

    Ref: https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/ssm/mamba_context_parallel.py#L324

    :param input_: The input tensor with shape ``[BT, H/CP, D]``, containing full sequence
        but partitioned along the head dimension.
    :param cp_group: The process group for context parallel communication.
    :returns: The output tensor with shape ``[BT/CP, H, D]``, partitioned along
        the sequence dimension but with full head dimension.
    """
    assert input_.dim() == 3, "all_to_all_hp2cp assumes 3-d input shape."
    world_size = get_world_size(cp_group)

    # Gather along middle dimension (for attention: [T, n_heads, head_dim])
    t_in, h_in, d_in = input_.shape
    t_out = t_in // world_size

    # [CP, t_out, h/CP, d]
    input_split = input_.reshape(world_size, t_out, h_in, d_in)
    flattened = input_split.flatten(0, 2)  # [CP*t_out*h/CP, d]
    exchanged = all_to_all(cp_group, flattened)

    output_split = exchanged.reshape(world_size, t_out, h_in, d_in)
    output = output_split.permute(1, 0, 2, 3).reshape(
        t_out, h_in * world_size, d_in
    )  # [t/CP, h, d]

    return output


class UlyssesContextParallelStyle(Config):
    """
    Configuration for Ulysses-style context parallelism.
    """

    pass


def _get_zig_zag_load_balancer():
    """Lazy import to avoid circular dependency."""
    from olmo_core.nn.attention.ring import RingAttentionLoadBalancerType

    return RingAttentionLoadBalancerType.zig_zag


@dataclass
class RingContextParallelStyle(Config):
    """
    Configuration for ring attention-style context parallelism.
    """

    load_balancer: "RingAttentionLoadBalancerType" = field(
        default_factory=_get_zig_zag_load_balancer
    )
    """
    The type of load balancer to use for ring attention.
    """

    head_stride: int = 1
    """
    The stride of the head dimension to process for each iteration of ring attention. A value of 1
    means each iteration will process one k and one v head. A value of 2 will process two k and two
    v heads, etc. A larger stride will reduce the number of communication ops.
    """
