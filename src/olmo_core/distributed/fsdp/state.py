from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List

import torch
from torch.utils.hooks import RemovableHandle

from olmo_core.stream import Stream
from olmo_core.utils import get_default_device

from .flat_param_handle import FlatParamHandle

if TYPE_CHECKING:
    from .fsdp import FSDP


@dataclass
class FSDPState:
    device: torch.device = field(default_factory=get_default_device)
    """
    The device the FSDP node is running on.
    """

    flat_param_handles: List[FlatParamHandle] = field(default_factory=list)
    """
    Manages the shared data for all sharded flat params.
    """

    pre_backward_hook_handles: List[RemovableHandle] = field(default_factory=list)
    """
    Backward hooks registered to the output tensors from the wrapped module's forward method.
    """

    post_backward_hook_handles: Dict[str, RemovableHandle] = field(default_factory=dict)
    """
    Post-backward hooks registered to the next autograd function in the graph for each parameter.
    The keys are parameter FQNs.
    """

    lazy_init_complete: bool = False
    """
    Marked true when final initialization runs lazily during the first forward pass.
    """

    params_prefetched: bool = False
    """
    Indicates that the unsharded params have already been prefetched.
    """

    forward_execution_order: List[FSDP] = field(default_factory=list)
    """
    The forward-pass execution order of all FSDP instances as determined by the first forward pass.
    This is used on subsequent steps to determine the prefetch order.
    """

    forward_execution_order_finalized: bool = False
    """
    Marked true when the forward pass execution order has been finalized after the first forward pass.
    """

    forward_prefetch_queue: deque[FSDP] = field(default_factory=lambda: deque([]))
    """
    Queue of FSDP modules to prefetch for unsharding during forward pass.
    """

    backward_execution_order: List[FSDP] = field(default_factory=list)
    """
    The backward-pass execution order of all FSDP instances as determined by the first backward pass.
    This is used on subsequent steps to determine the prefetch order.
    """

    backward_execution_order_finalized: bool = False
    """
    Marked true when the backward pass execution order has been finalized after the first backward pass.
    """

    backward_prefetch_queue: deque[FSDP] = field(default_factory=lambda: deque([]))
    """
    Queue of FSDP modules to prefetch for unsharding during backward pass.
    """

    compute_stream: Stream = field(default_factory=Stream.default)
    """
    Default stream for computation.
    """

    pre_unshard_stream: Stream = field(default_factory=Stream.default)
    """
    Stream used to allocate unsharded tensors prior to the all-gather.
    """

    unshard_stream: Stream = field(default_factory=Stream.default)
    """
    Stream used during the all-gather for unsharding parameters.
    """

    post_backward_stream: Stream = field(default_factory=Stream.default)
    """
    Stream used during the post-backward hook to cast gradients in preparation for the all-gather.
    """

    reduce_stream: Stream = field(default_factory=Stream.default)
    """
    Stream used during the reduce-scatter for reducing gradients after the backward pass.
    """

    @property
    def current_stream(self) -> Stream:
        return Stream.current(self.device)
