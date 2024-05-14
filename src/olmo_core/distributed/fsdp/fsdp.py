from __future__ import annotations

import logging
import math
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import partial
from itertools import chain
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.autograd import Variable
from torch.distributed.device_mesh import DeviceMesh

from olmo_core.distributed.tensors import ShardedFlatParameter
from olmo_core.stream import Stream
from olmo_core.utils import (
    StrEnum,
    apply_to_tensors,
    gc_cuda,
    get_default_device,
    get_grad_norm,
)

from .flat_param_handle import FlatParamHandle
from .state import FSDPState

log = logging.getLogger(__name__)


@dataclass
class FSDPPrecision:
    """
    Mixed precision settings for :class:`FSDP`.
    """

    param_dtype: Optional[torch.dtype] = None
    """
    The data type to cast full model parameters to during the forward and backward pass.
    """

    reduce_dtype: Optional[torch.dtype] = None
    """
    The data type used when reducing gradients. If not set this defaults to ``param_dtype``.
    """


class FSDPShardingStrategy(StrEnum):
    """
    Defines the sharding strategy used by :class:`FSDP`.
    """

    FULL_SHARD = "FULL_SHARD"
    """
    Parameters, gradients, and optimizer states are sharded. For the parameters, this strategy unshards
    (via all-gather) before the forward, reshards after the forward (except potentially for the root FSDP instance),
    unshards before the backward computation, and reshards after the backward computation.
    For gradients, it synchronizes and shards them (via reduce-scatter) after the backward
    computation. The sharded optimizer states are updated locally per rank.
    """

    HYBRID_SHARD = "HYBRID_SHARD"
    """
    Apply ``FULL_SHARD`` within a process group, and replicate parameters across process groups.
    This results in reduced communication volume as expensive all-gathers and reduce-scatters are
    only done within a node, which can be more performant for medium to large-sized models.
    """

    SHARD_GRAD_OP = "SHARD_GRAD_OP"
    """
    Like ``FULL_SHARD`` except parameters are not resharded after the forward pass when gradients
    are enabled, instead only after the backwards pass.
    """


@dataclass
class FSDPDebugConfig:
    no_reduce_grads: bool = False


M = TypeVar("M", bound=nn.Module)

ModuleWrapSpec = Union[Sequence[Union[str, nn.Module, Type[nn.Module]]], Callable[[nn.Module], bool]]


class FSDP(Generic[M], nn.Module):
    """
    FSDP, a.k.a. Fully Sharded Data Parallel, a ZeRO-3 model wrapper.

    :param module: The module to wrap.
    :param process_group: The distributed process group to shard across.
    :param device_mesh: Mutually exclusive with ``process_group``.
        This is required for :data:`FSDPShardingStrategy.HYBRID_SHARD`, in which case the first
        dimension should specify the number of model replicas (hybrid groups), and the second
        dimension should specify the number of shards within each replica.
        If you're not using :data:`FSDPShardingStrategy.HYBRID_SHARD` and you specify ``device_mesh``,
        the process group in the first dimension will be used.
    :param precision: Mixed precision settings.
    :param sharding_strategy: The sharding strategy to use.
    :param max_prefetch_count: The number of nested FSDP modules that can be prefetched during the forward
        and backward passes. This is like PyTorch's ``limit_all_gathers`` except it allows more control.
    :param free_root_after_forward: By default the root FSDP instance keeps its full params in memory after
        the forward pass when grads are enabled to avoid immediately regathering during the backward
        pass. Setting this to ``False`` can save some memory at the expense of throughput.
    """

    WRAPPED_MODULE_PREFIX = "_fsdp_wrapped_module"
    """
    The prefix the wrapped module is stored under. In general you don't need to know this as the wrapping
    FSDP instance behaves like the wrapped module itself for most APIs, and otherwise you should
    access the wrapped module through the :data:`module` property.
    """

    def __init__(
        self,
        module: M,
        process_group: Optional[dist.ProcessGroup] = None,
        device_mesh: Optional[DeviceMesh] = None,
        precision: Optional[FSDPPrecision] = None,
        sharding_strategy: FSDPShardingStrategy = FSDPShardingStrategy.FULL_SHARD,
        max_prefetch_count: int = 1,
        free_root_after_forward: bool = False,
        _debug_config: Optional[FSDPDebugConfig] = None,
    ):
        super().__init__()

        # Validate process group and device mesh given the sharding strategy.
        inter_group_process_group: Optional[dist.ProcessGroup] = None
        if process_group is not None and device_mesh is not None:
            raise ValueError("'process_group' and 'device_mesh' are mutually exclusive")
        elif device_mesh is not None:
            if sharding_strategy == FSDPShardingStrategy.HYBRID_SHARD:
                inter_group_process_group = cast(dist.ProcessGroup, device_mesh.get_group(mesh_dim=0))
                process_group = cast(dist.ProcessGroup, device_mesh.get_group(mesh_dim=1))
            else:
                process_group = cast(dist.ProcessGroup, device_mesh.get_group(mesh_dim=0))
        elif sharding_strategy == FSDPShardingStrategy.HYBRID_SHARD:
            raise ValueError("'device_mesh' is required for `HYBRID_SHARD`")

        self._fsdp_wrapped_module = module
        self.process_group = process_group
        self.inter_group_process_group = inter_group_process_group
        self.precision = precision or FSDPPrecision()
        self.sharding_strategy = sharding_strategy
        self.max_prefetch_count = max_prefetch_count
        self.free_root_after_forward = free_root_after_forward
        self.debug_config = _debug_config or FSDPDebugConfig()
        self.device = get_default_device()
        self.state = FSDPState(device=self.device)
        self.is_root = True

        # Shard the module in place.
        self._shard()

        # Mark all children as not root.
        for fsdp_child in self._fsdp_children(recurse=True):
            fsdp_child.is_root = False

    ################
    ## Public API ##
    ################

    @classmethod
    def auto_wrap(cls, module: M, children_to_wrap: ModuleWrapSpec, **fsdp_kwargs) -> FSDP[M]:
        """
        Wrap a module and specific children of the module specific by ``children_to_wrap``.

        :param children_to_wrap: Specify which children modules to wrap. This can be a list of children
            FQNs (wildcards allowed), module instances, module types, or a function that takes a
            module and returns a boolean that indicates whether it should be wrapped.
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
            if callable(children_to_wrap):
                should_wrap = children_to_wrap(child)
            else:
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
        """
        Run the forward pass on the wrapped module, gathering full parameters when necessary.
        """
        self._lazy_init()

        if self.is_root and self.state.forward_execution_order_finalized:
            # Fill forward-pass prefetch queue for unsharding.
            for module in self.state.forward_execution_order:
                self.state.forward_prefetch_queue.append(module)

        # Determine whether to reshard after the forward pass.
        keep_full_params_with_grads = (
            (self.is_root and not self.free_root_after_forward)
            or self.sharding_strategy == FSDPShardingStrategy.SHARD_GRAD_OP
        ) and torch.is_grad_enabled()

        # Unshard parameters in-place.
        self._unshard(set_grads=keep_full_params_with_grads)

        try:
            # Wait for unsharding stream before running the wrapped module's forward pass.
            self.state.compute_stream.wait_stream(self.state.unshard_stream)

            # Then we can prefetch the next FSDP module(s) asynchronously.
            if self.state.forward_execution_order_finalized:
                self._prefetch(self.state.forward_prefetch_queue)

            # Run forward pass on the wrapped module.
            with self.state.compute_stream:
                log.debug("Running forward pass for %s...", self.module.__class__.__name__)
                output = self.module(*args, **kwargs)

            if torch.is_grad_enabled():
                # Prepare for backward pass.
                if self.is_root and self.state.backward_execution_order_finalized:
                    # Fill backward-pass prefetch queue for unsharding.
                    for module in self.state.backward_execution_order:
                        self.state.backward_prefetch_queue.append(module)

                # If gradients are required, register a backward hook on the outputs to unshard
                # parameters in place again when needed.
                self._register_pre_backward_hooks(output)

                # Register post-backward hooks to reshard the parameters in place and reduce gradients.
                self._register_post_backward_hooks()
        finally:
            # Reshard parameters in-place, except potentially the root instance to avoid
            # immediately regathering in the backward pass.
            if not keep_full_params_with_grads:
                self._reshard()

        if self.is_root:
            # At the end of the first forward pass, execution order is now finalized, meaning
            # we can use 'self.state.forward_execution_order' to start prefetching unshards during
            # the next forward pass.
            if not self.state.forward_execution_order_finalized:
                self.state.forward_execution_order_finalized = True
                for child in self._fsdp_children(recurse=True):
                    child.state.forward_execution_order_finalized = True

            if self.state.forward_prefetch_queue:
                raise RuntimeError(
                    "Forward prefetch queue has not been emptied!\n"
                    f"Still contains {len(self.state.forward_prefetch_queue)} modules:\n"
                    f"{[m.module.__class__.__name__ for m in self.state.forward_prefetch_queue]}"
                )

        return output

    def state_dict(self, *args, **kwargs):
        """
        Return the state dict.

        .. seealso::
            For saving and loading :class:`FSDP` checkpoints, see :mod:`olmo_core.distributed.checkpoint`.

        .. tip::
            The data in the state dict will be sharded flat data unless you're within the :meth:`summon_full_params()`
            context or have gathered the full parameters another way.

        .. tip::
            The parameter names will be the original parameter names of the wrapped module, i.e.
            without the :data:`WRAPPED_MODULE_PREFIX`.
        """
        return self.module.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Load a state dict. The data in the state dict should correspond to the current state of the
        FSDP wrapper, either sharded or unsharded.

        .. seealso::
            For saving and loading :class:`FSDP` checkpoints, see :mod:`olmo_core.distributed.checkpoint`.
        """
        # Fix keys to include the right prefix.
        key_mapping = self._get_key_mapping()  # maps original key to wrapped key
        return super().load_state_dict({key_mapping.get(k, k): v for k, v in state_dict.items()}, *args, **kwargs)

    def named_buffers(self, *args, **kwargs):
        """
        Return an iterator over module buffers, yielding both the name of the buffer and the buffer itself.

        .. tip::
            The parameter names will be the original parameter names of the wrapped module, i.e.
            without the :data:`WRAPPED_MODULE_PREFIX`.
        """
        key_mapping = self._get_key_mapping(reverse=True)
        for name, buffer in super().named_buffers(*args, **kwargs):
            yield key_mapping.get(name, name), buffer

    def named_parameters(self, *args, **kwargs):
        """
        Return an iterator over module parameters, yielding both the name of the parameter as well
        as the parameter itself.

        .. tip::
            The parameter names will be the original parameter names of the wrapped module, i.e.
            without the :data:`WRAPPED_MODULE_PREFIX`.
        """
        key_mapping = self._get_key_mapping(reverse=True)
        for name, param in super().named_parameters(*args, **kwargs):
            yield key_mapping.get(name, name), param

    @contextmanager
    def summon_full_params(
        self, recurse: bool = True, writeback: bool = True, rank0_only: bool = False, cast: bool = False
    ):
        """
        Gather full unsharded params in-place with this context manager.

        :param recurse: Gather unsharded params for all child FSDP instances as well.
        :param writeback: Write the unsharded data back from rank 0 to all other ranks while exiting
            the context manager.
        :param rank0_only: Only summon full params on rank 0.
        :param cast: If using a mixed-precision strategy, params are cast to the same dtype as they
            are during the forward and backward passes. If this is ``True``, ``writeback`` must be
            ``False``.
        """
        if cast and writeback:
            raise ValueError("`summon_full_params` with `cast=True` and `writeback=True` is not supported")
        self._unshard(cast=cast, recurse=recurse, rank0_only=rank0_only)
        self.state.current_stream.wait_stream(self.state.unshard_stream)
        try:
            yield self
        finally:
            self._reshard(writeback=writeback, recurse=recurse)

    def apply(self, fn):
        """
        Apply ``fn`` recursively to every submodule (as returned by ``.children()``) as well as self.

        Typical use includes initializing the parameters of a model.

        Compared to :meth:`torch.nn.Module.apply`, this version additionally gathers the full parameters
        for all sharded parameters that are *directly managed* but the given FSDP instance before applying ``fn``.
        This should not be called from within another :meth:`summon_full_params()` context.
        """
        with self.summon_full_params(recurse=False, writeback=True, rank0_only=False):
            ret = super().apply(fn)

        return ret

    @torch.no_grad()
    def clip_grad_norm_(self, max_norm: float, norm_type: float = 2.0) -> torch.Tensor:
        """
        Clip the gradient norm of all parameters, returning the norm prior to clipping.

        The norm is computed over all parameters’ gradients as viewed as a single vector, and the
        gradients are modified in-place.
        """
        if not self.is_root:
            raise RuntimeError("`clip_grad_norm_()` should only be called on the root FSDP instance")

        sharded_params: Set[ShardedFlatParameter] = set()
        nonsharded_params: Set[nn.Parameter] = set()
        grads: List[torch.Tensor] = []
        for param in self.parameters():
            if param.grad is None or param.grad.numel() == 0:
                continue

            if isinstance(param, ShardedFlatParameter):
                sharded_params.add(param)
            else:
                nonsharded_params.add(param)
            grads.append(param.grad)

        if not grads:
            raise RuntimeError("`clip_grad_norm_()` was called but there are no gradients to clip!")

        local_sharded_norm = get_grad_norm(sharded_params, norm_type).to(self.device)
        global_nonsharded_norm = get_grad_norm(nonsharded_params, norm_type).to(self.device)

        # Reconstruct total gradient norm.
        total_norm: torch.Tensor
        if norm_type == math.inf:
            total_norm = torch.maximum(local_sharded_norm, global_nonsharded_norm)
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=self.process_group)
        else:
            total_norm = local_sharded_norm**norm_type
            dist.all_reduce(total_norm, group=self.process_group)
            total_norm += global_nonsharded_norm**norm_type
            total_norm = total_norm ** (1.0 / norm_type)

        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
        # Multiplying by the clamped coefficient is meaningless when it is
        # equal to 1, but it avoids the host-device sync that would result from
        # `if clip_coef < 1`
        for grad in grads:
            grad.detach().mul_(clip_coef_clamped.to(grad.device, grad.dtype))

        return total_norm

    def __getattr__(self, name: str) -> Any:
        """
        Forward missing attributes to the wrapped module.
        """
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._fsdp_wrapped_module, name)

    def __getitem__(self, key) -> Any:
        """
        Forward indexing calls in case the module is an ``nn.Sequential`` or ``nn.ModuleDict``.
        """
        if hasattr(self, FSDP.WRAPPED_MODULE_PREFIX):
            return self._fsdp_wrapped_module.__getitem__(key)  # type: ignore[operator]
        return super().__getitem__(key)  # type: ignore

    ##################
    ## Internal API ##
    ##################

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

        log.debug(
            "Completing lazy initialization from root FSDP for %s (%s)...",
            self.module.__class__.__name__,
            id(self.module),
        )

        # Initialize streams.
        self.state.compute_stream = Stream.default(self.device)
        self.state.pre_unshard_stream = Stream.new(self.device)
        self.state.unshard_stream = Stream.new(self.device)
        self.state.post_backward_stream = Stream.new(self.device)
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
                pre_unshard_stream=self.state.pre_unshard_stream,
                unshard_stream=self.state.unshard_stream,
                post_backward_stream=self.state.post_backward_stream,
                reduce_stream=self.state.reduce_stream,
                forward_execution_order=self.state.forward_execution_order,
                forward_prefetch_queue=self.state.forward_prefetch_queue,
                backward_execution_order=self.state.backward_execution_order,
                backward_prefetch_queue=self.state.backward_prefetch_queue,
            )

    def _get_key_mapping(self, reverse: bool = False, modules: bool = False) -> Dict[str, str]:
        """
        Get mapping of original keys to wrapped keys, or the other way around if ``reverse=True``.
        """
        key_mapping: Dict[str, str] = {}  # maps original key to wrapped key

        def collect_key_mappings(module: nn.Module, og_prefix: str, wrapped_prefix: str):
            if isinstance(module, FSDP):
                wrapped_prefix = f"{wrapped_prefix}{self.WRAPPED_MODULE_PREFIX}."
                module = module.module

            if not modules:
                for param_name, _ in chain(
                    module.named_parameters(recurse=False), module.named_buffers(recurse=False)
                ):
                    key_mapping[f"{og_prefix}{param_name}"] = f"{wrapped_prefix}{param_name}"

            for child_name, child in module.named_children():
                if modules:
                    key_mapping[og_prefix.strip(".")] = wrapped_prefix.strip(".")
                collect_key_mappings(child, f"{og_prefix}{child_name}.", f"{wrapped_prefix}{child_name}.")

        collect_key_mappings(self.module, "", f"{self.WRAPPED_MODULE_PREFIX}.")

        if reverse:
            key_mapping = {v: k for k, v in key_mapping.items()}

        return key_mapping

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

    def _managed_named_parameters(self) -> Generator[Tuple[str, ShardedFlatParameter], None, None]:
        """
        Returns a generator over all parameters directly managed by this FSDP instance.
        """
        for handle in self.state.flat_param_handles:
            for param_name, param in zip(handle.param_fqns, handle.params):
                yield param_name, param

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
        Shard the wrapped module in place, replacing each ``nn.Parameter`` with a ``ShardedFlatParameter``,
        and then collecting all sharded flat param data into a single ``FlatParamHandle``. Afterwards
        the sharded data in each sharded flat param will be a view into a single flat tensor managed
        by the flat param handle.

        This should only be called once at initialization.
        """
        log.debug("Sharding %s (%s)...", self.module.__class__.__name__, id(self.module))

        params_with_grads: List[nn.Parameter] = []
        params_with_grads_fqns: List[str] = []
        params_without_grads: List[nn.Parameter] = []
        params_without_grads_fqns: List[str] = []

        # NOTE: this generator will include `self.module` itself
        for module_name, module in self._named_children(recurse=lambda m: not isinstance(m, FSDP)):
            if isinstance(module, FSDP):
                continue
            for param_name, param in module.named_parameters(recurse=False):
                param_fqn = f"{module_name}.{param_name}"
                if param.requires_grad:
                    params_with_grads.append(param)
                    params_with_grads_fqns.append(param_fqn)
                else:
                    params_without_grads.append(param)
                    params_without_grads_fqns.append(param_fqn)

        # Collate the data from params into the flat param handle. The data in each flat param
        # will then just be a view into a slice of the data managed by the flat param handle.
        # This makes unsharding more efficient as we'll only need a single `all_gather` call.
        handles = []
        if params_with_grads:
            handles.append(
                FlatParamHandle.shard_params(
                    params_with_grads,
                    params_with_grads_fqns,
                    process_group=self.process_group,
                    inter_group_process_group=self.inter_group_process_group,
                    device=self.device,
                )
            )
        if params_without_grads:
            handles.append(
                FlatParamHandle.shard_params(
                    params_without_grads,
                    params_without_grads_fqns,
                    process_group=self.process_group,
                    inter_group_process_group=self.inter_group_process_group,
                    device=self.device,
                )
            )

        self.state.flat_param_handles = handles

        for module_name, module in self._named_children(recurse=lambda m: not isinstance(m, FSDP)):
            if isinstance(module, FSDP):
                continue
            for param_name, param in module.named_parameters(recurse=False):
                param_fqn = f"{module_name}.{param_name}"
                for handle in handles:
                    try:
                        idx_in_handle = handle.param_fqns.index(param_fqn)
                    except ValueError:
                        continue
                    sharded_flat_param = handle.params[idx_in_handle]
                    setattr(module, param_name, sharded_flat_param)
                    break

        gc_cuda()

    @torch.no_grad()
    def _unshard(
        self,
        cast: bool = True,
        set_grads: bool = False,
        recurse: bool = False,
        rank0_only: bool = False,
    ):
        """
        Unshard the wrapped module in place.
        """
        if self.state.params_prefetched:
            return

        kwargs = dict(cast=cast, set_grads=set_grads, recurse=recurse, rank0_only=rank0_only)

        log.debug("Unsharding %s (%s)...", self.module.__class__.__name__, id(self.module))
        self.state.params_prefetched = True

        # NOTE: `unshard_stream` should wait on current stream (usually `compute_stream` / `default_stream`)
        # if root to respect the optimizer step and any other computations on the params outside of this
        # module's forward/backward pass.
        if self.is_root:
            self.state.unshard_stream.wait_stream(self.state.current_stream)

        with self.state.pre_unshard_stream:
            for handle in self.state.flat_param_handles:
                handle.pre_unshard_(
                    dtype=self.precision.param_dtype if cast else None,
                    rank0_only=rank0_only,
                    set_grads=set_grads,
                )

        with self.state.unshard_stream(wait_stream=self.state.pre_unshard_stream):
            for handle in self.state.flat_param_handles:
                handle.unshard_(
                    dtype=self.precision.param_dtype if cast else None,
                    rank0_only=rank0_only,
                    set_grads=set_grads,
                )

        # Record the current stream for the unsharded parameter data to make sure it's not
        # deallocated prematurely.
        for handle in self.state.flat_param_handles:
            self.state.current_stream.record_for(handle.params_data)
            if handle.params_unsharded_grad is not None:
                self.state.current_stream.record_for(handle.params_unsharded_grad)

        if recurse:
            for module in self._fsdp_children():
                module._unshard(**kwargs)

    def _prefetch(self, prefetch_from: deque[FSDP], **kwargs):
        for module in self._deque_from(prefetch_from):
            log.debug(
                "Prefetching %s (%s) from %s (%s)...",
                module.module.__class__.__name__,
                id(module.module),
                self.module.__class__.__name__,
                id(self.module),
            )
            module._unshard(**kwargs)

    @torch.no_grad()
    def _reshard(self, writeback: bool = False, recurse: bool = False):
        """
        Re-shard the wrapped module in place. Should be called after :meth:`unshard()`.
        """
        kwargs = dict(writeback=writeback, recurse=recurse)

        log.debug("Resharding %s (%s)...", self.module.__class__.__name__, id(self.module))
        self.state.params_prefetched = False

        for handle in self.state.flat_param_handles:
            handle.reshard_(writeback=writeback)

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

        log.debug("Reduce-scattering grads for %s (%s)", self.module.__class__.__name__, id(self.module))
        grad_reduce_dtype: Optional[torch.dtype] = self.precision.reduce_dtype or self.precision.param_dtype

        with self.state.post_backward_stream(wait_stream=self.state.current_stream):
            for handle in self.state.flat_param_handles:
                handle.pre_reduce_scatter_grads_(grad_reduce_dtype=grad_reduce_dtype)

        with self.state.reduce_stream(wait_stream=self.state.post_backward_stream):
            for handle in self.state.flat_param_handles:
                handle.reduce_scatter_grads_(grad_reduce_dtype=grad_reduce_dtype)

        with self.state.post_backward_stream(wait_stream=self.state.reduce_stream):
            for handle in self.state.flat_param_handles:
                handle.post_reduce_scatter_grads_(grad_reduce_dtype=grad_reduce_dtype)

    def _deque_from(self, prefetch_queue: deque[FSDP]) -> Generator[FSDP, None, None]:
        count = 0
        while prefetch_queue and count < self.max_prefetch_count:
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
        log.debug("Running pre-backward hook for %s (%s)...", self.module.__class__.__name__, id(self.module))

        # Remove all pre backward hooks for this FSDP instance since they all do the same thing.
        for handle in self.state.pre_backward_hook_handles:
            handle.remove()
        self.state.pre_backward_hook_handles.clear()

        if self.is_root:
            self._register_post_backward_final_hook()

        # Unshard parameters in place.
        self._unshard(set_grads=True)

        # Wait for unshard stream so gradient computation can proceed.
        self.state.current_stream.wait_stream(self.state.unshard_stream)

        if self.state.backward_execution_order_finalized:
            # Prefetch next FSDP module(s) asynchronously.
            self._prefetch(self.state.backward_prefetch_queue, set_grads=True)
        else:
            # Add self to backward execution order.
            self.state.backward_execution_order.append(self)

    def _register_pre_backward_hook(self, x: torch.Tensor):
        if x.requires_grad:
            hook_handle = x.register_hook(self._pre_backward_hook)
            self.state.pre_backward_hook_handles.append(hook_handle)

    def _register_pre_backward_hooks(self, output: Any):
        log.debug("Registering pre-backward hooks for %s (%s)...", self.module.__class__.__name__, id(self.module))
        # Clear existing hooks if there are any.
        if self.state.pre_backward_hook_handles:
            log.debug(
                "Removing old pre-backward hooks for %s (%s)...", self.module.__class__.__name__, id(self.module)
            )
            for hook_handle in self.state.pre_backward_hook_handles:
                hook_handle.remove()
            self.state.pre_backward_hook_handles.clear()
        apply_to_tensors(self._register_pre_backward_hook, output)

    ### Post-backward hook to reshard parameters in-place and reduce-scatter gradients across
    ### the process group. Also accumulates any cached sharded gradients.

    @torch.no_grad()
    def _post_backward_hook(self, param_name: str, *unused: Any):
        del unused
        self.state.post_backward_hook_handles.pop(param_name).remove()

        # If there are still more handles then there are still more post-backward hooks to be ran
        # in the current FSDP node. Only the last handle should do the work.
        if self.state.post_backward_hook_handles:
            return

        log.debug("Running post-backward hook for %s (%s)...", self.module.__class__.__name__, id(self.module))

        # NOTE: reshard *before* reducing grads to correctly handle precision settings.
        self._reshard()
        self._reduce_scatter_grads()

    def _register_post_backward_hook(self, param_name: str, param: ShardedFlatParameter):
        # Force creation of a `grad_fn` in order to register a hook that will run *after* this param's
        # backward pass.
        tmp_param = param.expand_as(param)
        assert tmp_param.grad_fn is not None
        acc_grad = tmp_param.grad_fn.next_functions[0][0]
        assert acc_grad is not None
        handle = acc_grad.register_hook(partial(self._post_backward_hook, param_name))
        self.state.post_backward_hook_handles[param_name] = handle

    def _register_post_backward_hooks(self):
        log.debug(
            "Registering post-backward hooks for %s (%s)...", self.module.__class__.__name__, id(self.module)
        )
        # Clear existing hooks if there are any.
        if self.state.post_backward_hook_handles:
            log.debug(
                "Removing old post-backward hooks for %s (%s)...", self.module.__class__.__name__, id(self.module)
            )
            for handle in self.state.post_backward_hook_handles.values():
                handle.remove()
            self.state.post_backward_hook_handles.clear()
        for param_name, param in self._managed_named_parameters():
            if param.requires_grad:
                self._register_post_backward_hook(param_name, param)

    @torch.no_grad()
    def _post_backward_final_hook(self):
        if not self.is_root:
            return

        log.debug("Running post-backward final hook for %s (%s)", self.module.__class__.__name__, id(self.module))

        # Mark backward execution order as finalized.
        self.state.backward_execution_order_finalized = True
        for child in self._fsdp_children(recurse=True):
            child.state.backward_execution_order_finalized = True

        # Wait for unsharding and reducing streams to complete so the model is not left in a bad
        # state before grad clipping, optimizer step, or whatever else.
        self.state.current_stream.wait_stream(self.state.reduce_stream)

    def _register_post_backward_final_hook(self):
        if not self.is_root:
            return

        log.debug(
            "Registering post-backward final hook for %s (%s)...", self.module.__class__.__name__, id(self.module)
        )
        Variable._execution_engine.queue_callback(self._post_backward_final_hook)
