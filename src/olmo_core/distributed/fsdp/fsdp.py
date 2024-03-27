from __future__ import annotations

import logging
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import partial
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import torch
import torch.distributed as dist
import torch.nn as nn

from olmo_core.distributed.sharded_flat_parameter import ShardedFlatParameter
from olmo_core.utils import apply_to_tensors, get_default_device

from .state import FSDPState
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
class FSDPDebugConfig:
    no_reduce_grads: bool = False


M = TypeVar("M", bound=nn.Module)

ModuleWrapSpec = Sequence[Union[str, nn.Module, Type[nn.Module]]]


class FSDP(Generic[M], nn.Module):
    """
    This is a complete rewrite of PyTorch's ``FullyShardedDataParallel``, a ZeRO-3 model wrapper.

    :param module: The module to wrap.
    :param process_group: The distributed process group.
    :param precision: Mixed precision settings.
    :param max_prefetch_count: The number of nested FSDP modules that can be prefetched during the forward
        and backward passes. This is like PyTorch's ``limit_all_gathers`` except it allows more control.
    """

    WRAPPED_MODULE_PREFIX = "_fsdp_wrapped_module"

    def __init__(
        self,
        module: M,
        process_group: Optional[dist.ProcessGroup] = None,
        precision: Optional[FSDPPrecision] = None,
        max_prefetch_count: int = 1,
        _debug_config: Optional[FSDPDebugConfig] = None,
    ):
        super().__init__()
        setattr(self, FSDP.WRAPPED_MODULE_PREFIX, module)
        self.process_group = process_group
        self.precision = precision or FSDPPrecision()
        self.max_prefetch_count = max_prefetch_count
        self.debug_config = _debug_config or FSDPDebugConfig()
        self.device = get_default_device()
        self.state = FSDPState(device=self.device)
        self.is_root = True

        # Shard the module in place.
        self._shard()

        # Mark all children as not root.
        for fsdp_child in self._fsdp_children(recurse=True):
            fsdp_child.is_root = False

    @classmethod
    def auto_wrap(cls, module: M, children_to_wrap: ModuleWrapSpec, **fsdp_kwargs) -> FSDP[M]:
        """
        Wrap a module and specific children of the module specific by ``children_to_wrap``.

        :param children_to_wrap: Specify which children modules to wrap. This can be a list of children
            FQNs (wildcards allowed), a list of module instances, or a list of module types.
        :param fsdp_kwargs: Keyword args to the FSDP constructor.
        """
        from fnmatch import fnmatch

        def named_modules_with_parent(
            parent: nn.Module, parent_fqn: str
        ) -> Generator[Tuple[nn.Module, nn.Module, str, str], None, None]:
            for child_name, child_module in parent.named_children():
                child_fqn = f"{parent_fqn}.{child_name}" if parent_fqn else child_name
                yield parent, child_module, child_fqn, child_name
                yield from named_modules_with_parent(child_module, child_fqn)

        for parent, child, child_fqn, child_name in named_modules_with_parent(module, ""):
            should_wrap = False
            for wrap_spec in children_to_wrap:
                if isinstance(wrap_spec, str):
                    should_wrap = child_fqn == wrap_spec or fnmatch(child_fqn, wrap_spec)
                elif isinstance(wrap_spec, nn.Module):
                    should_wrap = child is wrap_spec
                elif issubclass(wrap_spec, nn.Module):
                    should_wrap = isinstance(child, wrap_spec)
                else:
                    raise TypeError(f"unexpected type in 'children_to_wrap' ({type(wrap_spec)})")

                if should_wrap:
                    break

            if should_wrap:
                setattr(parent, child_name, cls(child, **fsdp_kwargs))

        return cls(module, **fsdp_kwargs)

    @property
    def module(self) -> M:
        """
        Get the wrapped module.
        """
        return self._fsdp_wrapped_module

    def forward(self, *args, **kwargs):
        self._lazy_init()

        log.debug("Running forward pass for %s...", self.module.__class__.__name__)

        if self.is_root and self.state.forward_execution_order_finalized:
            # Fill forward-pass prefetch queue for unsharding.
            for module in self.state.forward_execution_order:
                self.state.forward_prefetch_queue.append(module)

        # Unshard parameters in-place.
        self._unshard(
            prefetch_from=self.state.forward_prefetch_queue
            if self.state.forward_execution_order_finalized
            else None
        )

        try:
            # Run forward pass on the original model.
            with self.state.compute_stream(wait_stream=self.state.unshard_stream):
                output = self.module(*args, **kwargs)
        finally:
            # Reshard parameters in-place.
            self._reshard()

        if self.is_root:
            # At the end of the first forward pass, execution order is now finalized, meaning
            # we can use 'self.state.forward_execution_order' to start prefetching unshards.
            self.state.forward_execution_order_finalized = True
            assert not self.state.forward_prefetch_queue

        if torch.is_grad_enabled():
            if self.is_root and self.state.backward_execution_order_finalized:
                # Fill backward-pass prefetch queue for unsharding.
                for module in self.state.backward_execution_order:
                    self.state.backward_prefetch_queue.append(module)

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
        self.state.current_stream.wait_stream(self.state.unshard_stream)
        try:
            yield self
        finally:
            self._reshard(writeback=writeback, recurse=recurse)
            self.state.current_stream.wait_stream(self.state.unshard_stream)

    def apply(self, fn):
        """
        Apply ``fn`` recursively to every submodule (as returned by ``.children()``) as well as self.

        Typical use includes initializing the parameters of a model.

        Compared to ``torch.nn.Module.apply``, this version additionally gathers the full parameters
        for all sharded parameters that are *directly managed* but the given FSDP instance before applying ``fn``.
        This should not be called from within another ``summon_full_params`` context.
        """
        with self.summon_full_params(recurse=False, writeback=True, rank0_only=False):
            ret = super().apply(fn)

        return ret

    def _lazy_init(self):
        """
        Complete initialization of streams and other stuff.
        Should be called automatically during first forward pass.
        """
        if self.state.lazy_init_complete:
            return

        self.state.lazy_init_complete = True
        if not self.is_root:
            # Mark 'self' next in the execution order.
            assert self.state.forward_execution_order
            self.state.forward_execution_order.append(self)
            return

        log.debug("Completing lazy initialization from root FSDP for %s...", self.module.__class__.__name__)

        # Initialize streams.
        self.state.compute_stream = Stream.default(self.device)
        self.state.unshard_stream = Stream.new(self.device)
        self.state.reduce_stream = Stream.new(self.device)

        # Initialize execution order.
        self.state.forward_execution_order.clear()
        self.state.forward_execution_order.append(self)
        self.state.backward_execution_order.clear()

        for fsdp_child in self._fsdp_children(recurse=True):
            # Set child to use same streams.
            fsdp_child.state = replace(
                fsdp_child.state,
                compute_stream=self.state.compute_stream,
                unshard_stream=self.state.unshard_stream,
                reduce_stream=self.state.reduce_stream,
                forward_execution_order=self.state.forward_execution_order,
                forward_prefetch_queue=self.state.forward_prefetch_queue,
                backward_execution_order=self.state.backward_execution_order,
                backward_prefetch_queue=self.state.backward_prefetch_queue,
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
        for _, m in self._named_children(
            recurse=lambda m: not isinstance(m, FSDP)
        ):  # NOTE: this generator will include `self.module` itself
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
        self,
        cast: bool = True,
        cache_grads: bool = False,
        recurse: bool = False,
        rank0_only: bool = False,
        prefetch_from: Optional[deque[FSDP]] = None,
    ):
        """
        Unshard the wrapped module in place.
        """
        if self.state.params_prefetched:
            return

        kwargs = dict(cast=cast, cache_grads=cache_grads, recurse=recurse, rank0_only=rank0_only)

        log.debug("Unsharding %s...", self.module.__class__.__name__)
        self.state.params_prefetched = True

        # NOTE: `unshard_stream` should wait on current stream (usually `compute_stream` / `default_stream`)
        # if root to respect the optimizer step and any other computations on the params outside of this
        # module's forward/backward pass.
        with self.state.unshard_stream(wait_stream=self.state.current_stream if self.is_root else None):
            # TODO: batch the unshards for all params together?
            for param_name, param in self._managed_named_parameters():
                if not isinstance(param, ShardedFlatParameter):
                    continue

                param.unshard_(dtype=self.precision.param_dtype if cast else None, rank0_only=rank0_only)
                if cache_grads and param.grad is not None:
                    # We should only be caching these between the pre-backward and post-backward
                    # hooks. The post-backward hook will remove the cached grad as it accumulates
                    # it into persistent sharded grad.
                    assert param_name not in self.state.sharded_grad_cache
                    self.state.sharded_grad_cache[param_name] = param.grad.detach()
                    param.grad = None

        if prefetch_from is not None:
            for module in self._deque_from(prefetch_from):
                log.debug(
                    "Prefetching %s from %s...", module.module.__class__.__name__, self.module.__class__.__name__
                )
                module._unshard(**kwargs)

        if recurse:
            for module in self._fsdp_children():
                module._unshard(**kwargs)

    @torch.no_grad()
    def _reshard(self, writeback: bool = False, recurse: bool = False):
        """
        Re-shard the wrapped module in place. Should be called after :meth:`unshard()`.
        """
        kwargs = dict(writeback=writeback, recurse=recurse)

        log.debug("Resharding %s...", self.module.__class__.__name__)
        self.state.params_prefetched = False

        with self.state.unshard_stream(wait_stream=self.state.compute_stream):
            # TODO: batch the unshards for all params together?
            for _, param in self._managed_named_parameters():
                if not isinstance(param, ShardedFlatParameter):
                    continue

                param.reshard_(writeback=writeback)

        if recurse:
            for module in self._fsdp_children():
                module._reshard(**kwargs)

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

                if (cached_grad := self.state.sharded_grad_cache.pop(param_name, None)) is not None:
                    param.grad.add_(cached_grad)
                    del cached_grad

    def _deque_from(self, prefetch_queue: deque[FSDP]) -> Generator[FSDP, None, None]:
        count = 0
        while prefetch_queue and count <= self.max_prefetch_count:
            module = prefetch_queue.popleft()
            if module is not self:
                count += 1
                yield module

    ###########
    ## Hooks ##
    ###########

    ### Pre-backward hook to unshard parameters in-place and cache existing sharded grads for
    ### gradient accumulation.

    @torch.no_grad()
    def _pre_backward_hook(self, *unused: Any):
        del unused
        log.debug("Running pre-backward hook for %s...", self.module.__class__.__name__)

        if not self.state.backward_execution_order_finalized:
            # Add self to backward execution order.
            self.state.backward_execution_order.append(self)

        # Unshard parameters in place.
        self._unshard(
            cache_grads=True,
            prefetch_from=self.state.backward_prefetch_queue
            if self.state.backward_execution_order_finalized
            else None,
        )

        # Remove all pre backward hooks for this FSDP instance since they all do the same thing.
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

        # If there are still more handles then there are still more post-backward hooks to be ran
        # in the current FSDP node. Only the last handle should do the work.
        if self.state.post_backward_hook_handles:
            return

        # NOTE: reshard *before* reducing grads to correctly handle precision settings.
        # '_reduce_scatter_grads' checks 'param.dtype' to determine dtype for grads, which
        # at that point should be the original dtype.
        self._reshard()
        self._reduce_scatter_grads()

        # The root FSDP instance needs to do some final cleanup.
        if not self.is_root:
            return

        # Mark backward execution order as finalized.
        self.state.backward_execution_order_finalized = True

        # Wait for unsharding and reducing streams to complete so the model is not left in a bad
        # state before grad clipping, optimizer step, or whatever else.
        self.state.current_stream.wait_stream(self.state.unshard_stream)
        self.state.current_stream.wait_stream(self.state.reduce_stream)

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
