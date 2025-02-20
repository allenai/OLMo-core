from dataclasses import dataclass
from typing import List, Set

import torch
from torch.distributed import DeviceMesh

from olmo_core.config import Config


@dataclass
class ContextParallelConfig(Config):
    """
    Configuration class for context parallelism (CP).
    """

    degree: int
    """
    The CP degree.
    """


def create_context_parallel_ctx(
    *,
    cp_mesh: DeviceMesh,
    cp_buffers: List[torch.Tensor],
    cp_seq_dims: List[int],
    cp_no_restore_buffers: Set[torch.Tensor],
    cp_rotate_method: str,
):
    from torch.distributed.tensor.experimental import context_parallel
    from torch.distributed.tensor.experimental._attention import set_rotate_method

    set_rotate_method(cp_rotate_method)
    return context_parallel(
        cp_mesh,
        buffers=cp_buffers,
        buffer_seq_dims=cp_seq_dims,
        no_restore_buffers=cp_no_restore_buffers,
    )
