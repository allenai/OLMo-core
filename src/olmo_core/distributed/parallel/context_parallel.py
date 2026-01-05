from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from olmo_core.config import Config
from olmo_core.distributed.nn import all_to_all_single
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


def all_to_all_hp2cp(
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
    assert input_.dim() == 4, "all_to_all_hp2cp expects 4-d input shape [B, T, H/CP, D]."
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


class UlyssesContextParallelStyle(Config):
    """
    Configuration for Ulysses-style context parallelism.
    """

    @property
    def load_balancer(self) -> "RingAttentionLoadBalancerType":
        return RingAttentionLoadBalancerType.ulysses


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
