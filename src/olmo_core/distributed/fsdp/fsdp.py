import logging
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, Generic, List, Optional, TypeVar

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from ...utils import apply_to_tensors, get_default_device
from ..sharded_flat_parameter import ShardedFlatParameter

log = logging.getLogger(__name__)


@dataclass
class FSDPPrecision:
    param_dtype: Optional[torch.dtype] = None
    """
    The type for model parameters during the forward and backward pass.
    """

    reduce_dtype: Optional[torch.dtype] = None
    """
    The data type used when reducing gradients.
    """


@dataclass
class FSDPState:
    pre_backward_hook_handles: List[RemovableHandle] = field(default_factory=list)
    """
    Backward hooks registered to the output tensors from the wrapped module's forward method.
    """

    post_backward_hook_handles: Dict[str, RemovableHandle] = field(default_factory=dict)


M = TypeVar("M", bound=nn.Module)


class FSDP(Generic[M], nn.Module):
    def __init__(
        self,
        module: M,
        process_group: Optional[dist.ProcessGroup] = None,
        precision: Optional[FSDPPrecision] = None,
    ):
        super().__init__()
        self._fsdp_wrapped_module = module
        self.process_group = process_group
        self.precision = precision or FSDPPrecision()
        self.state = FSDPState()
        self.device = get_default_device()

        # Shard the module in place.
        self._shard()

    @property
    def module(self) -> M:
        """
        Get the wrapped module.
        """
        return self._fsdp_wrapped_module

    def forward(self, *args, **kwargs):
        log.debug("Running forward pass for %s...", self.module.__class__.__name__)

        # Unshard parameters in-place.
        self._unshard()

        try:
            # Run forward pass on the original model.
            output = self.module(*args, **kwargs)
        finally:
            # Reshard parameters in-place.
            self._reshard()

        if torch.is_grad_enabled():
            # If gradients are required, register a backward hook on the outputs to unshard
            # parameters in place again when needed.
            self._register_pre_backward_hooks(output)

            # And post-backward hooks to reshard the parameters in place and reduce gradients.
            self._register_post_backward_hooks()

        return output

    @torch.no_grad()
    def _shard(self):
        """
        Shard the wrapped module in place. This should only be called once.
        """
        log.debug("Sharding %s...", self.module.__class__.__name__)
        for m in self.module.modules():
            for param_name, param in m.named_parameters(recurse=False):
                # TODO: use better sharding strategy that doesn't potentially always result in highest rank with
                # smallest shard.
                sharded_flat_param = ShardedFlatParameter.shard(
                    param, process_group=self.process_group, device=self.device
                )
                setattr(m, param_name, sharded_flat_param)

    @torch.no_grad()
    def _unshard(self):
        """
        Unshard the wrapped module in place.
        """
        log.debug("Unsharding %s...", self.module.__class__.__name__)
        for m in self.module.modules():
            for param in m.parameters(recurse=False):
                if isinstance(param, ShardedFlatParameter):
                    param.unshard_(dtype=self.precision.param_dtype)

    @torch.no_grad()
    def _reshard(self):
        """
        Re-shard the wrapped module in place. Should be called after :meth:`unshard()`.
        """
        log.debug("Resharding %s...", self.module.__class__.__name__)
        for param in self.module.parameters():
            if isinstance(param, ShardedFlatParameter):
                param.reshard_()

    @torch.no_grad()
    def _reduce_scatter_grads(self):
        """
        Reduce and scatter unsharded gradients across the process group, leaving only sharded
        gradients in their place.
        """
        for param_name, param in self.module.named_parameters():
            if (unsharded_grad := param.grad) is None:
                continue

            log.debug("Reduce-scattering grads for %s.%s...", self.module.__class__.__name__, param_name)

            if not isinstance(param, ShardedFlatParameter):
                dist.all_reduce(unsharded_grad, group=self.process_group)
                param.grad = unsharded_grad
                continue

            # Only NCCL supports 'reduce_scatter'. So with other backends we use 'all_reduce'.
            if dist.get_backend() == dist.Backend.NCCL:
                # Get chunks corresponding to each rank.
                grad_chunks = param.chunk_unsharded(unsharded_grad, pad=True)
                new_sharded_grad = torch.empty_like(grad_chunks[0])
                dist.reduce_scatter(new_sharded_grad, grad_chunks, group=self.process_group)
                param.grad = new_sharded_grad
            else:
                dist.all_reduce(unsharded_grad, group=self.process_group)
                param.grad = param.sharded_chunk(unsharded_grad)

    ###########
    ## Hooks ##
    ###########

    ### Pre-backward ###

    @torch.no_grad()
    def _pre_backward_hook(self, *unused: Any):
        del unused
        log.debug("Running pre-backward hook for %s...", self.module.__class__.__name__)
        # Unshard parameters in place.
        self._unshard()
        # Remove all pre backward hooks since they all do the same thing.
        for handle in self.state.pre_backward_hook_handles:
            handle.remove()
        self.state.pre_backward_hook_handles.clear()

    def _register_pre_backward_hook(self, x: torch.Tensor):
        handle = x.register_hook(self._pre_backward_hook)
        self.state.pre_backward_hook_handles.append(handle)

    def _register_pre_backward_hooks(self, output: Any):
        log.debug("Registering pre-backward hooks for %s...", self.module.__class__.__name__)
        # Clear existing hooks if there are any.
        if self.state.pre_backward_hook_handles:
            log.debug("Removing old pre-backward hooks for %s...", self.module.__class__.__name__)
            for handle in self.state.pre_backward_hook_handles:
                handle.remove()
            self.state.pre_backward_hook_handles.clear()
        apply_to_tensors(self._register_pre_backward_hook, output)

    ### Post-backward ###

    @torch.no_grad()
    def _post_backward_hook(self, param_name: str, *unused: Any):
        del unused
        log.debug("Running post-backward hook for %s.%s...", self.module.__class__.__name__, param_name)
        self.state.post_backward_hook_handles.pop(param_name).remove()
        if not self.state.post_backward_hook_handles:
            self._reshard()
            self._reduce_scatter_grads()

    def _register_post_backward_hook(self, param_name: str, param: nn.Parameter):
        if isinstance(param, ShardedFlatParameter) and param.requires_grad:
            # Force creation of a `grad_fn` in order to register a hook that will after this param's
            # backward pass.
            tmp_param = param.expand_as(param)
            assert tmp_param.grad_fn is not None
            acc_grad = tmp_param.grad_fn.next_functions[0][0]
            assert acc_grad is not None
            handle = acc_grad.register_hook(partial(self._post_backward_hook, param_name))
            self.state.post_backward_hook_handles[param_name] = handle

    def _register_post_backward_hooks(self):
        log.debug("Registering post-backward hooks for %s...", self.module.__class__.__name__)
        # Clear existing hooks if there are any.
        if self.state.post_backward_hook_handles:
            log.debug("Removing old post-backward hooks for %s...", self.module.__class__.__name__)
            for handle in self.state.post_backward_hook_handles.values():
                handle.remove()
            self.state.post_backward_hook_handles.clear()
        for param_name, param in self.module.named_parameters():
            self._register_post_backward_hook(param_name, param)
