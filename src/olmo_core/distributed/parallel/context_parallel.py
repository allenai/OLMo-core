import contextlib
from dataclasses import dataclass
from typing import Generator, List, Set

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


_CONTEXT_PARALLEL_ENABLED: bool = False


@contextlib.contextmanager
def context_parallel_manager(
    *,
    cp_mesh: DeviceMesh,
    cp_buffers: List[torch.Tensor],
    cp_seq_dims: List[int],
    cp_no_restore_buffers: Set[torch.Tensor],
) -> Generator[None, None, None]:
    """
    A context manager for enabling context parallelism.
    """
    from torch.distributed.tensor.experimental import context_parallel
    from torch.nn.attention import SDPBackend, sdpa_kernel

    global _CONTEXT_PARALLEL_ENABLED

    with contextlib.ExitStack() as stack:
        # Currently ring attention only supports these two SDP backends.
        stack.enter_context(
            sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
        )

        # Patch SDPA and shard these buffers on their sequence dimension.
        stack.enter_context(
            context_parallel(
                cp_mesh,
                buffers=cp_buffers,
                buffer_seq_dims=cp_seq_dims,
                no_restore_buffers=cp_no_restore_buffers,
            )
        )

        try:
            _CONTEXT_PARALLEL_ENABLED = True
            yield
        finally:
            _CONTEXT_PARALLEL_ENABLED = False


def context_parallel_enabled() -> bool:
    """
    Indicates if context parallelism is currently enabled with :func:`context_parallel_manager()`.
    """
    return _CONTEXT_PARALLEL_ENABLED
