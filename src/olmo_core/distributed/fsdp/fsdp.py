from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.hooks import RemovableHandle

from olmo_core.distributed.sharded_flat_parameter import ShardedFlatParameter
from olmo_core.utils import apply_to_tensors, get_default_device

from .stream import Stream

log = logging.getLogger(__name__)


@dataclass
class FSDPPrecision:
    param_dtype: Optional[torch.dtype] = None
    """
    The data type to cast full model parameters to during the forward and backward pass.
    """

    reduce_dtype: Optional[torch.dtype] = None
    """
    The data type used when reducing gradients. If not set this defaults to ``param_dtype``.
    """

    keep_low_precision_grads: bool = False
    """
    If ``True``, gradients are kept in low precision (if ``param_dtype`` is to set to low precision)
    instead of being upcast to full precision for the optimizer.
    """


@dataclass
class FSDPState:
    device: torch.device = field(default_factory=get_default_device)

    pre_backward_hook_handles: List[RemovableHandle] = field(default_factory=list)
    """
    Backward hooks registered to the output tensors from the wrapped module's forward method.
    """

    post_backward_hook_handles: Dict[str, RemovableHandle] = field(default_factory=dict)
    """
    Post-backward hooks registered to the next autograd function in the graph for each parameter.
    The keys are parameter FQNs.
    """

    compute_stream: Stream = field(default_factory=Stream.default)
    """
    Default stream for computation.
    """

    unshard_stream: Stream = field(default_factory=Stream.default)
    """
    Stream for unsharding parameters.
    """

    reduce_stream: Stream = field(default_factory=Stream.default)
    """
    Stream for reducing gradients after the backward pass.
    """

    def wait_for_compute_stream(self):
        """
        Has unshard stream wait for computation stream.
        For example, this should be called in the root FSDP instance pre-forward to respect
        optimizer step computation.
        """
        self.unshard_stream.wait_stream(self.compute_stream)

    @property
    def current_stream(self) -> Stream:
        return Stream.current(self.device)


@dataclass
class FSDPDebugConfig:
    no_reduce_grads: bool = False


M = TypeVar("M", bound=nn.Module)


class FSDP(Generic[M], nn.Module):
    WRAPPED_MODULE_PREFIX = "_fsdp_wrapped_module"

    def __init__(
        self,
        module: M,
        process_group: Optional[dist.ProcessGroup] = None,
        precision: Optional[FSDPPrecision] = None,
        _debug_config: Optional[FSDPDebugConfig] = None,
    ):
        super().__init__()
        setattr(self, FSDP.WRAPPED_MODULE_PREFIX, module)
        self.process_group = process_group
        self.precision = precision or FSDPPrecision()
        self.debug_config = _debug_config or FSDPDebugConfig()
        self.device = get_default_device()
        self.state = FSDPState(device=self.device)
        self.is_root = True
        # For caching sharded gradients during gradient accumulation.
        # Maps param FQN to local sharded gradient.
        self._sharded_grad_cache: Dict[str, torch.Tensor] = {}
        self._lazy_init_complete = False

        # Shard the module in place.
        self._shard()

        # Mark all children as not root.
        for fsdp_child in self._fsdp_children(recurse=True):
            fsdp_child.is_root = False

    @property
    def module(self) -> M:
        """
        Get the wrapped module.
        """
        return self._fsdp_wrapped_module

    def forward(self, *args, **kwargs):
        self._lazy_init()

        if self.is_root:
            self.state.wait_for_compute_stream()

        log.debug("Running forward pass for %s...", self.module.__class__.__name__)

        # Unshard parameters in-place.
        # TODO: figure out how to prefetch
        self._unshard()

        try:
            # Run forward pass on the original model.
            with self.state.compute_stream(wait_stream=self.state.unshard_stream):
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

    def state_dict(self, *args, **kwargs):
        """
        Return the state dict. The keys in the state dict will always correspond to the original keys
        in the wrapped model.

        The data in the state dict will be sharded flat data unless you're within the :meth:`summon_full_params()`
        context or have gathered the full parameters another way.
        """
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Load a state dict. The data in the state dict should correspond to the current state of the
        FSDP wrapper, either sharded or unsharded.
        """
        # Fix keys to include the right prefix.
        key_mapping: Dict[str, str] = {}  # maps original key to wrapped key

        def collect_key_mappings(module: nn.Module, og_prefix: str, wrapped_prefix: str):
            for param_name, _ in module.named_parameters(recurse=False):
                key_mapping[f"{og_prefix}{param_name}"] = f"{wrapped_prefix}{param_name}"

            if isinstance(module, FSDP):
                wrapped_prefix = f"{wrapped_prefix}{self.WRAPPED_MODULE_PREFIX}."
                module = module.module

            for child_name, child in module.named_children():
                collect_key_mappings(child, f"{og_prefix}{child_name}.", f"{wrapped_prefix}{child_name}.")

        collect_key_mappings(self.module, "", f"{self.WRAPPED_MODULE_PREFIX}.")

        return super().load_state_dict({key_mapping.get(k, k): v for k, v in state_dict.items()}, *args, **kwargs)

    @contextmanager
    def summon_full_params(self, recurse: bool = True, writeback: bool = True, rank0_only: bool = False):
        """
        Gather full unsharded params in-place with this context manager.

        :param recurse: Gather unsharded params for all child FSDP instances as well.
        :param writeback: Write the unsharded data back from rank 0 to all other ranks while exiting
            the context manager.
        :param rank0_only: Only summon full params on rank 0.
        """
        self._unshard(cast=False, recurse=recurse, rank0_only=rank0_only)
        try:
            yield self
        finally:
            self._reshard(writeback=writeback, recurse=recurse)

    def _lazy_init(self):
        """
        Complete initialization of streams and other stuff.
        """
        if self._lazy_init_complete:
            return

        self._lazy_init_complete = True
        if not self.is_root:
            return False

        log.debug("Completing lazy initialization from root FSDP for %s...", self.module.__class__.__name__)

        # Initialize streams.
        self.state.compute_stream = Stream.default(self.device)
        self.state.unshard_stream = Stream.new(self.device)
        self.state.reduce_stream = Stream.new(self.device)

        for fsdp_child in self._fsdp_children(recurse=True):
            fsdp_child._lazy_init_complete = True  # not really necessary

            # Set child to use same streams.
            fsdp_child.state = replace(
                fsdp_child.state,
                compute_stream=self.state.compute_stream,
                unshard_stream=self.state.unshard_stream,
                reduce_stream=self.state.reduce_stream,
            )

    def _named_children(
        self, recurse: Union[bool, Callable[[nn.Module], bool]] = True
    ) -> Generator[Tuple[str, nn.Module], None, None]:
        """
        Returns a generator over children modules with their names, only recursing further if the condition is met.
        """

        def collect_children(module: nn.Module, prefix: str = "") -> Generator[Tuple[str, nn.Module], None, None]:
            for child_name, child in module.named_children():
                yield prefix + child_name, child
                if recurse is True or (callable(recurse) and recurse(module)):
                    yield from collect_children(child, prefix=f"{prefix}{child_name}.")

        yield from collect_children(self.module)

    def _managed_named_parameters(self) -> Generator[Tuple[str, nn.Parameter], None, None]:
        """
        Returns a generator over all parameters managed by this FSDP instance. This is equivalent
        to `self.module.named_parameters()` except that parameters within nested FSDP instances are omitted.
        """
        for module_name, module in self._named_children(recurse=lambda m: not isinstance(m, FSDP)):
            if not isinstance(module, FSDP):
                for param_name, param in module.named_parameters(recurse=False):
                    yield f"{module_name}.{param_name}", param

    def _fsdp_children(self, recurse: bool = False) -> Generator[FSDP, None, None]:
        """
        Returns a generator over all child FSDP instances of this module.

        :recurse: Whether to recurse into each FSDP child.
        """
        for _, module in self._named_children(recurse=recurse or (lambda m: not isinstance(m, FSDP))):
            if isinstance(module, FSDP):
                yield module

    @torch.no_grad()
    def _shard(self):
        """
        Shard the wrapped module in place, replacing each ``nn.Parameter`` with a ``ShardedFlatParameter``.
        This should only be called once at initialization.
        """
        log.debug("Sharding %s...", self.module.__class__.__name__)
        for m in self.module.modules():
            if isinstance(m, FSDP):
                continue
            for param_name, param in m.named_parameters(recurse=False):
                # TODO: use better sharding strategy that doesn't potentially always result in highest rank with
                # smallest shard.
                sharded_flat_param = ShardedFlatParameter.shard(
                    param, process_group=self.process_group, device=self.device, synchronize=False
                )
                setattr(m, param_name, sharded_flat_param)

    @torch.no_grad()
    def _unshard(
        self, cast: bool = True, cache_grads: bool = False, recurse: bool = False, rank0_only: bool = False
    ):
        """
        Unshard the wrapped module in place.
        """
        log.debug("Unsharding %s...", self.module.__class__.__name__)

        # TODO: figure out how to limit the number of all gathers here like PyTorch FSDP does
        # when you set 'limit_all_gathers=True'.
        with self.state.unshard_stream:
            # TODO: batch the unshards for all params together?
            for param_name, param in self._managed_named_parameters():
                if not isinstance(param, ShardedFlatParameter):
                    continue

                param.unshard_(dtype=self.precision.param_dtype if cast else None, rank0_only=rank0_only)
                if cache_grads and param.grad is not None:
                    # We should only be caching these between the pre-backward and post-backward
                    # hooks. The post-backward hook will remove the cached grad as it accumulates
                    # it into persistent sharded grad.
                    assert param_name not in self._sharded_grad_cache
                    self._sharded_grad_cache[param_name] = param.grad.detach()
                    param.grad = None

        if recurse:
            for module in self._fsdp_children():
                module._unshard(cast=cast, cache_grads=cache_grads, recurse=recurse, rank0_only=rank0_only)

    @torch.no_grad()
    def _reshard(self, writeback: bool = False, recurse: bool = False):
        """
        Re-shard the wrapped module in place. Should be called after :meth:`unshard()`.
        """
        log.debug("Resharding %s...", self.module.__class__.__name__)

        with self.state.unshard_stream:
            # TODO: batch the unshards for all params together?
            for _, param in self._managed_named_parameters():
                if not isinstance(param, ShardedFlatParameter):
                    continue

                param.reshard_(writeback=writeback)

        if recurse:
            for module in self._fsdp_children():
                module._reshard(writeback=writeback, recurse=recurse)

    @torch.no_grad()
    def _reduce_scatter_grads(self):
        """
        Reduce and scatter unsharded gradients across the process group, leaving only sharded
        gradients in their place. This also checks for cached sharded gradients
        (cached during gradient accumulation) and accumulates those before clearing that cache.
        """
        if self.debug_config.no_reduce_grads:
            log.warning(
                "Skipping reduce-scattering grads for %s due to debug config.",
                self.module.__class__.__name__,
            )
            return

        # dtype to keep sharded gradients in.
        grad_dtype: Optional[torch.dtype] = (
            self.precision.param_dtype if self.precision.keep_low_precision_grads else None
        )
        # dtype just for reducing gradients.
        grad_reduce_dtype: Optional[torch.dtype] = self.precision.reduce_dtype or self.precision.param_dtype

        with self.state.reduce_stream(wait_stream=self.state.current_stream):
            # TODO: batch the reductions for all params together?
            for param_name, param in self._managed_named_parameters():
                if (unsharded_grad := param.grad) is None:
                    continue

                log.debug("Reduce-scattering grads for %s.%s...", self.module.__class__.__name__, param_name)

                if grad_reduce_dtype is not None:
                    unsharded_grad = unsharded_grad.to(dtype=grad_reduce_dtype)

                if not isinstance(param, ShardedFlatParameter):
                    dist.all_reduce(unsharded_grad, group=self.process_group)
                    param.grad = unsharded_grad
                    continue

                if grad_dtype is None:
                    grad_dtype = param.dtype

                # Only NCCL supports 'reduce_scatter'. So with other backends we use 'all_reduce'.
                if dist.get_backend() == dist.Backend.NCCL:
                    # Get chunks corresponding to each rank.
                    grad_chunks = param.chunk_unsharded(unsharded_grad, pad=True)
                    new_sharded_grad = torch.empty_like(grad_chunks[0])
                    dist.reduce_scatter(new_sharded_grad, grad_chunks, group=self.process_group)
                    param.grad = new_sharded_grad[: param.unsharded_flattened_offsets[1]].to(dtype=grad_dtype)
                else:
                    dist.all_reduce(unsharded_grad, group=self.process_group)
                    param.grad = param.sharded_chunk(unsharded_grad).detach().to(dtype=grad_dtype)

                del unsharded_grad

                if (cached_grad := self._sharded_grad_cache.pop(param_name, None)) is not None:
                    param.grad.add_(cached_grad)
                    del cached_grad

    ###########
    ## Hooks ##
    ###########

    ### Pre-backward hook to unshard parameters in-place and cache existing sharded grads for
    ### gradient accumulation.

    @torch.no_grad()
    def _pre_backward_hook(self, *unused: Any):
        del unused
        log.debug("Running pre-backward hook for %s...", self.module.__class__.__name__)

        # Unshard parameters in place.
        # TODO: figure out how to prefetch
        self._unshard(cast=True, cache_grads=True)

        # Remove all pre backward hooks since they all do the same thing.
        for handle in self.state.pre_backward_hook_handles:
            handle.remove()
        self.state.pre_backward_hook_handles.clear()

        # Wait for unshard stream so gradient computation can proceed.
        self.state.current_stream.wait_stream(self.state.unshard_stream)

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

    ### Post-backward hook to reshard parameters in-place and reduce-scatter gradients across
    ### the process group. Also accumulates any cached sharded gradients.

    @torch.no_grad()
    def _post_backward_hook(self, param_name: str, *unused: Any):
        del unused
        log.debug("Running post-backward hook for %s.%s...", self.module.__class__.__name__, param_name)
        self.state.post_backward_hook_handles.pop(param_name).remove()
        if not self.state.post_backward_hook_handles:
            # NOTE: reshard *before* reducing grads to correctly handle precision settings.
            # '_reduce_scatter_grads' checks 'param.dtype' to determine dtype for grads, which
            # at that point should be the original dtype.
            self._reshard()
            self._reduce_scatter_grads()

    def _register_post_backward_hook(self, param_name: str, param: nn.Parameter):
        if isinstance(param, ShardedFlatParameter) and param.requires_grad:
            # Force creation of a `grad_fn` in order to register a hook that will run *after* this param's
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
        for param_name, param in self._managed_named_parameters():
            self._register_post_backward_hook(param_name, param)
