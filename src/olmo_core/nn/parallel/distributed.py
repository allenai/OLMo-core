# mypy: allow-untyped-defs
import copy
import functools
import inspect
import itertools
import logging
import os
import sys
import warnings
import weakref
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, fields, is_dataclass
from enum import auto, Enum
from typing import Any, Callable, List, Optional, TYPE_CHECKING, Dict

import torch
import torch.distributed as dist
import torch.nn as nn
from torch._utils import _get_device_index
from torch.autograd import Function, Variable
from torch.distributed.algorithms.join import Join, Joinable, JoinHook
from torch.nn.modules import Module
from torch.nn.parallel.scatter_gather import gather, scatter_kwargs
from torch.utils._pytree import tree_flatten, tree_unflatten
from collections import OrderedDict

RPC_AVAILABLE = False
if dist.is_available():
    from torch.distributed.distributed_c10d import (
        _get_default_group,
        _rank_not_in_group,
        ReduceOp,
    )
    from torch.distributed.utils import (
        _alloc_storage,
        _cast_forward_inputs,
        _free_storage,
        _sync_module_states,
        _to_kwargs,
        _verify_param_shape_across_processes,
    )
if dist.rpc.is_available():
    RPC_AVAILABLE = True
    from torch.distributed.rpc import RRef

if TYPE_CHECKING:
    from torch.utils.hooks import RemovableHandle


__all__ = ["DistributedDataParallel"]

logger = logging.getLogger(__name__)


def _find_tensors(obj):
    r"""Recursively find all tensors contained in the specified object."""
    if RPC_AVAILABLE and isinstance(obj, RRef):
        # If the current node is the owner of the RRef, unwrap it and try to
        # find Tensors.
        # TODO: Expand to remote RRefs.
        if obj.is_owner():
            return _find_tensors(obj.local_value())
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return itertools.chain.from_iterable(map(_find_tensors, obj))
    if isinstance(obj, dict):
        return itertools.chain.from_iterable(map(_find_tensors, obj.values()))
    if is_dataclass(obj):
        return itertools.chain.from_iterable(
            map(_find_tensors, (getattr(obj, f.name) for f in fields(obj)))
        )

    return []



class _BufferCommHookLocation(Enum):
    PRE_FORWARD = auto()
    POST_FORWARD = auto()


@dataclass
class _BufferCommHook:
    buffer_comm_hook: Callable
    buffer_comm_hook_state: Any
    buffer_comm_hook_location: _BufferCommHookLocation

@dataclass
class _ProcessGroupReducerState:
    process_group: Any
    parameters: list[torch.Tensor]
    expect_sparse_gradient: list[bool]
    reducer: Optional[dist.Reducer]
    param_to_name_mapping: dict[int, str]



class MultiGroupDistributedDataParallel(Module):
    

    def __init__(
        self,
        module,
        # device_ids=None,
        # output_device=None,
        dim=0,
        broadcast_buffers=True,
        init_sync=True,
        process_group=None,
        bucket_cap_mb=None,
        # find_unused_parameters=False,
        # gradient_as_bucket_view=False,
        # static_graph=False,
        # delay_all_reduce_named_params=None,
        # param_to_hook_all_reduce=None,
        # mixed_precision: Optional[_MixedPrecision] = None,
        # device_mesh=None,
        # skip_all_reduce_unused_params=False,
        ####### 
        param_process_group_fn=None,  # NEW
        accumulate_grads_in_fp32=False,  # NEW
        reduce_grads_in_fp32=False,  # NEW
    ):
        super().__init__()
        # Joinable.__init__(self)
        self._use_python_reducer = (
            torch._dynamo.utils.get_optimize_ddp_mode() == "python_reducer"
        )
        if not self._use_python_reducer:
             assert False, "Only python_reducer is supported. Please set torch._dynamo.config.optimize_ddp = \"python_reducer\""
        self.logger: Optional[dist.Logger] = None

        if process_group is None:
            self.process_group = _get_default_group()

        if hasattr(module, "_ddp_params_and_buffers_to_ignore"):
            self.parameters_to_ignore = set(module._ddp_params_and_buffers_to_ignore)
        else:
            self.parameters_to_ignore = set()

        self._module_parameters = [
            p
            for n, p in module.named_parameters()
            if n not in self.parameters_to_ignore
        ]

        # this is the order to launch grad reduce
        self._reversed_module_parameters = list(reversed(self._module_parameters))

        if not any(p.requires_grad for p in self._module_parameters):
            raise (
                RuntimeError,
                "DistributedDataParallel is not needed when a module "
                "doesn't have any parameter that requires a gradient.",
            )

        is_multi_device_module = (
            len({p.device for p in self._module_parameters}) > 1
        )
        if is_multi_device_module:
            raise NotImplementedError(
                "DistributedDataParallel with parameters on multiple devices is not supported yet."
            )
        distinct_device_types = {
            p.device.type for p in self._module_parameters if p.device is not None
        }

        self._debug_mode = True
        self.device_type = next(iter(distinct_device_types))

        self.static_graph = False
        self.dim = dim
        self.module = module
        self.device = next(iter(self._module_parameters)).device
        self.broadcast_buffers = broadcast_buffers
        self.require_backward_grad_sync = True
        self.require_forward_param_sync = True
        self.overlap_grad_reduce = True

        # Multi-process-group support.
        if param_process_group_fn is None:
            param_process_group_fn = lambda param: self.process_group # default to single process group 
        self._param_process_group_fn = param_process_group_fn


        self._accumulate_grads_in_fp32 = accumulate_grads_in_fp32
        self._reduce_grads_in_fp32 = reduce_grads_in_fp32

        if self._accumulate_grads_in_fp32 and not self._reduce_grads_in_fp32:
            raise ValueError(
                "accumulate_grads_in_fp32 requires reduce_grads_in_fp32 to be True"
            )


        # Check that a module does not have Uninitialized parameters
        for param in self._module_parameters:
            if isinstance(param, torch.nn.parameter.UninitializedParameter):
                raise RuntimeError(
                    "Modules with uninitialized parameters can't be used with `DistributedDataParallel`. "
                    "Run a dummy forward pass to correctly initialize the modules",
                )
            # disable meta device parameters
            if param.device.type == "meta":
                raise RuntimeError(
                    "Modules with meta device parameters can't be used with `DistributedDataParallel`. "
                    "Please initialize all parameters before wrapping with DistributedDataParallel.",
                )
        # used for intra-node param sync and inter-node sync as well
        self.broadcast_bucket_size = int(250 * 1024 * 1024)

        # reduction bucket size
        if bucket_cap_mb is None:
            # default case (bucket cap is 25 MiB)
            bucket_cap_mb = 25
            self.bucket_bytes_cap_default = True
        else:
            self.bucket_bytes_cap_default = False
        self.bucket_bytes_cap = int(bucket_cap_mb * 1024 * 1024)

        # Whether to perform input tensor CPU to GPU copies on a side-stream
        self.use_side_stream_for_tensor_copies = (
            os.environ.get("PYTORCH_DDP_USE_SIDE_STREAM", "1") == "1"
        )

        # self.reduce_stream = torch.Stream()

        # build param <-> process group mapping
        self.process_group_to_params: Dict[Any, List[torch.nn.Parameter]] = {}
        self.param_to_process_group: Dict[torch.nn.Parameter, Any] = {}

        named_buffers = list(self.module.named_buffers())
        if len(named_buffers) > 0:
            raise NotImplementedError(
                "DDP with param_process_group_fn does not support buffers yet."
            )

        for name, param in self.module.named_parameters():
            if name in self.parameters_to_ignore:
                continue
            pg = self._param_process_group_fn(name, param)

            if pg is None:
                pg = self.process_group

            self.param_to_process_group[param] = pg
            if pg not in self.process_group_to_params:
                self.process_group_to_params[pg] = []
            self.process_group_to_params[pg].append(param)


        if init_sync:
            self.init_sync()
                

        self._comm_hooks: list[tuple[object, object]] = []

        self._fp32_acc_hooks = []
        if self._accumulate_grads_in_fp32:
            for p in module.parameters():
                if not p.requires_grad:
                    continue
                # persistent FP32 master grad on the same device
                p._main_grad_fp32 = None # type: ignore[attr-defined]

                self._fp32_acc_hooks.append(p.register_post_accumulate_grad_hook(_fp32_post_grad_acc_hook))

        self._grad_reduce_hooks = []

        # Register the AccumulateGrad post hooks if optimize_ddp is
        # True. The hooks will be deregistered if compiled_autograd is not
        # enabled.
        self._accum_grad_hooks: list[RemovableHandle] = []

        self._param_grad_ready = OrderedDict()
        self._next_reduce_ptr = 0

        if self._use_python_reducer:
            torch._inductor.config._fuse_ddp_communication = True
            torch._inductor.config._fuse_ddp_bucket_size = bucket_cap_mb
            # Directly adding this to the trace rule will disturb the users
            # who are using DDPOptimizer.
            torch._dynamo.trace_rules.LEGACY_MOD_INLINELIST.add(
                "torch.nn.parallel.distributed"
            )
            torch._dynamo.trace_rules.get_legacy_mod_inlinelist.cache_clear()

            # the hook that controls gradient allreduce
            self._register_accum_grad_hook()

        # Whether or not DDPSink performs a clone.
        self._ddp_sink_clone = True


    def __getattr__(self, name: str) -> Any:
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # nn.Module logic first
        except AttributeError:
            return getattr(self.module, name)

    def __getitem__(self, key: int) -> Any:
        return self.module.__getitem__(key)  # type: ignore[operator]

    
    def _reduce_grad_for_one_param(
            self, 
            param,
            *,
            param_index: int,
            process_group
        ):
        # print('compiled_accum_grad_hook called for param index:', param_index)

        if self._accumulate_grads_in_fp32:
            # check accumulated grad
            if not hasattr(param, "_main_grad_fp32") or param._main_grad_fp32 is None:
                print(f"[Warning] param index={param_index}, shape={param.shape} has no _main_grad_fp32")
                return
        else:
            # check regular grad
            if param.grad is None:
                print(f"[Warning] param index={param_index}, shape={param.shape} has no grad")
                return

        if self._comm_hooks:
            assert False
            for hook, state in self._comm_hooks:
                hook(state, (param.grad, param))
        else:
            if self._accumulate_grads_in_fp32:
                # use _main_grad_fp32
                assert self._reduce_grads_in_fp32, "reduce_grads_in_fp32 must be True when accumulate_grads_in_fp32 is True"

                # gradient = param._main_grad_fp32 / process_group.size()
                param._main_grad_fp32.div_(process_group.size())
                # gradient = fcol.all_reduce(gradient, "sum", self.process_group)
                # self.reduce_stream.wait_stream(torch.cuda.current_stream())
                # with torch.cuda.stream(self.reduce_stream):
                handle = torch.distributed.all_reduce(param._main_grad_fp32, op=ReduceOp.SUM, group=process_group, async_op=True)
                self._grad_reduce_hooks.append((handle, None, None)) # no need to copy

                # param._main_grad_fp32.copy_(gradient)
            else:
                # use param.grad (bfloat16)
                if self._reduce_grads_in_fp32:
                    gradient = param.grad.float() / process_group.size()
                    # self.reduce_stream.wait_stream(torch.cuda.current_stream())
                    # with torch.cuda.stream(self.reduce_stream):
                    handle = torch.distributed.all_reduce(gradient, op=ReduceOp.SUM, group=process_group, async_op=True)
                    self._grad_reduce_hooks.append((handle, param.grad, gradient)) # need to write back to bf16 grad

                else:
                    param.grad.div_(process_group.size())
                    # self.reduce_stream.wait_stream(torch.cuda.current_stream())
                    # with torch.cuda.stream(self.reduce_stream):
                    handle = torch.distributed.all_reduce(param.grad, op=ReduceOp.SUM, group=process_group, async_op=True)
                    self._grad_reduce_hooks.append((handle, None, None)) # no need to copy

    def _maybe_kick_start_all_reduce(self):
        all_params_in_reverse = self._reversed_module_parameters

        while self._next_reduce_ptr != len(all_params_in_reverse): # while not pointing to the end
            # if self._debug_mode:
            #     print(f"rank={dist.get_rank()} next_reduce_ptr={self._next_reduce_ptr}")
            # check if we can start to reduce the next param grad
            param_next = all_params_in_reverse[self._next_reduce_ptr]
            if not param_next.requires_grad:
                self._next_reduce_ptr +=1 # skip this one
            else:
                if self._param_grad_ready[param_next]:
                    # if the grad is ready, launch AR
                    self._reduce_grad_for_one_param(param=param_next, param_index=self._next_reduce_ptr, process_group=self.param_to_process_group[param_next])
                    self._next_reduce_ptr += 1
                else:
                    # stop here and wait for the grad to be ready.
                    break


    def _register_accum_grad_hook(self):
        import torch.distributed._functional_collectives as fcol

        def notify_grad_ready(
            param,
        ):
            if not self.require_backward_grad_sync:
                return
            self._param_grad_ready[param] = True

            # do this in backward
            if self.overlap_grad_reduce:
                self._maybe_kick_start_all_reduce()

            # otherwise, leave the all-reduce to finalize_grad_reduce
            

        for index, param in enumerate(self._module_parameters):
            if not param.requires_grad:
                continue

            # set up param order
            self._param_grad_ready[param] = False

            # NOTE: in order to ensure param grads reduce always happen in the same order,
            # instead of launching all-reduce in accumulate_grad_hook
            # it only notifies the grad is ready
            # and the actual all-reduce is kicked off in _maybe_kick_start_all_reduce
            # based on what grads are ready
            self._accum_grad_hooks.append(
                param.register_post_accumulate_grad_hook(
                    notify_grad_ready
                )
            )


    def finalize_grad_reduce(self):

        # in some cases (eg, imbalance moe routing), some params may not have grads, and their
        # post_accumulate_grad_hook is never called, so their grad_ready is never set to True.
        # 
        if self._next_reduce_ptr < len(self._module_parameters):
            for param in self._param_grad_ready.keys():
                if not self._param_grad_ready[param]:
                    self._param_grad_ready[param] = True
                    # make zero grad for those params
                    if self._accumulate_grads_in_fp32:
                        if not hasattr(param, "_main_grad_fp32") or param._main_grad_fp32 is None:
                            param._main_grad_fp32 = torch.zeros_like(param, dtype=torch.float32)
                    else:
                        if param.grad is None:
                            param.grad = torch.zeros_like(param)

            self._maybe_kick_start_all_reduce()

        # now all grad reduce should have been launched
        assert self._next_reduce_ptr == len(self._module_parameters), f"Not all all-reduce operations have been launched: {self._next_reduce_ptr} vs {len(self._module_parameters)}"


        for idx, (handle, target, source) in enumerate(self._grad_reduce_hooks):
            # print(f'wait {idx}')
            handle.wait()
            if target is not None: # if target and source are None, no need to copy
                assert source is not None # target and source should be both None or not None
                target.copy_(source.to(target.dtype))
        self._grad_reduce_hooks = []

        self._next_reduce_ptr = 0 # point to the start of the reversed param list
        
        # mark all grads as not ready
        for key in self._param_grad_ready.keys():
            self._param_grad_ready[key] = False


    def init_sync(self):
        for process_group, parameters in self.process_group_to_params.items():
            # Verify model equivalence.
            _verify_param_shape_across_processes(process_group, parameters)

            for param in parameters:
                dist.broadcast(
                    param.data, src=dist.get_global_rank(process_group, 0), group=process_group, async_op=False
                )


    def __getstate__(self):
        # TODO: review if this works with multi-process-group DDP
        raise NotImplementedError("DDP serialization is not implemented.")


    def __setstate__(self, state):
        # TODO: review if this works with multi-process-group DDP
        raise NotImplementedError("DDP serialization is not implemented.")
        # If serializable, then the process group should be the default one




    def _get_parameters(self, m, recurse=True):
        """Return a generator of module parameters."""

        def model_parameters(m):
            ps = (
                m._former_parameters.values()
                if hasattr(m, "_former_parameters")
                else m.parameters(recurse=False)
            )
            yield from ps

        for mod in m.modules() if recurse else [m]:
            yield from model_parameters(mod)

    def _check_default_group(self):
        pickle_not_supported = False
        try:
            if self.process_group != _get_default_group():
                pickle_not_supported = True
        except RuntimeError:
            pickle_not_supported = True

        if pickle_not_supported:
            self._log_and_throw(
                RuntimeError,
                "DDP Pickling/Unpickling are only supported "
                "when using DDP with the default process "
                "group. That is, when you have called "
                "init_process_group and have not passed "
                "process_group argument to DDP constructor",
            )

    @contextmanager
    def no_sync(self):
        r"""
        Context manager to disable gradient synchronizations across DDP processes.

        Within this context, gradients will be accumulated on module
        variables, which will later be synchronized in the first
        forward-backward pass exiting the context.

        Example::

            >>> # xdoctest: +SKIP("undefined variables")
            >>> ddp = torch.nn.parallel.DistributedDataParallel(model, pg)
            >>> with ddp.no_sync():
            >>>     for input in inputs:
            >>>         ddp(input).backward()  # no synchronization, accumulate grads
            >>> ddp(another_input).backward()  # synchronize grads

        .. warning::
            The forward pass should be included inside the context manager, or
            else gradients will still be synchronized.
        """
        old_require_backward_grad_sync = self.require_backward_grad_sync
        self.require_backward_grad_sync = False
        try:
            yield
        finally:
            self.require_backward_grad_sync = old_require_backward_grad_sync


    def _run_ddp_forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)  # type: ignore[index]

    def _clear_grad_buffer(self):
        # Making param.grad points to the grad buffers before backward is based on the
        # assumption that the grad accumulation is done in place in autograd engine,
        # for some edge cases, if the grad accumulation in autograd engine is not in
        # place, then the param.grad and grad buffers are detached.
        if self._delay_grad_buffer is not None:
            # We batch zero_grad for all params by resetting the whole grad
            # buffer when the grad of all params is set to None.
            all_param_grad_none = all(
                param.grad is None for param in self._delay_all_reduce_params
            )

            for index, param in enumerate(self._delay_all_reduce_params):
                if param.grad is None:
                    param.grad = self._delay_grad_views[index]
                    if not all_param_grad_none:
                        param.grad.zero_()

            if all_param_grad_none:
                self._delay_grad_buffer.zero_()



    def _pre_forward(self, *inputs, **kwargs):        
        return inputs, kwargs


    def _post_forward(self, output):
        return output


    def forward(self, *inputs, **kwargs):
        with torch.autograd.profiler.record_function("DistributedDataParallel.forward"):
            inputs, kwargs = self._pre_forward(*inputs, **kwargs)
            output = self._run_ddp_forward(*inputs, **kwargs)
            
            return self._post_forward(output)

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def to_kwargs(self, inputs, kwargs, device_id):
        # Kept for BC
        return _to_kwargs(
            inputs,
            kwargs,
            torch.device(self.device_type, device_id),
            self.use_side_stream_for_tensor_copies,
        )

    def gather(self, outputs, output_device):
        return gather(outputs, output_device, dim=self.dim)

    def train(self, mode=True):
        super().train(mode)
        return self



    # When running in join mode, checks and performs sync of module buffers if
    # the models have buffers that should be synchronized in the forward pass.
    def _check_and_sync_module_buffers(self):
        if self._check_sync_bufs_pre_fwd():
            authoritative_rank = self._find_common_rank(self._distributed_rank, False)
            self._sync_module_buffers(authoritative_rank)

    # When running in join model, agrees upon a common rank and broadcast model
    # parameters to all other ranks.
    def _sync_final_model(self, is_last_joiner):
        # Agree upon the process that will be the authoritative model copy.
        # The current rank is a candidate for being the authoritative copy if
        # is_last_joiner=True. We break ties via picking the larger rank.
        self._authoritative_rank = self._find_common_rank(
            self._distributed_rank, is_last_joiner
        )
        _sync_module_states(
            module=self.module,
            process_group=self.process_group,
            broadcast_bucket_size=self.broadcast_bucket_size,
            src=self._authoritative_rank,
            params_and_buffers_to_ignore=self.parameters_to_ignore,
            broadcast_buffers=self.broadcast_buffers,
        )

    # Schedule comm ops to match those scheduled in the reducer's backward
    # pass.
    def _match_all_reduce_for_bwd_pass(self):
        comm_work = []
        # Schedule comm in the same order as Reducer schedules them, i.e.
        # the order of the buckets. Retrieving the bucket order from the reducer
        # ensures that we keep the same order in join mode, such as when bucket
        # order is rebuilt dynamically.

        # Returns grad_buckets in order, but real tensors are substituted with
        # zero tensors of the same shape.
        grad_buckets = self.reducer._get_zeros_like_grad_buckets()
        for grad_bucket in grad_buckets:
            # Joined processes contribute zero gradient. In the case that
            # divide_by_initial_world_size=True, we divide grads by the static
            # world size, if not, the dividing factor is reduced by the number
            # of joined processes.
            work = self.reducer._run_comm_hook(grad_bucket)
            comm_work.append(work)
        for work in comm_work:
            work.wait()

    # Allreduces the used parameter mapping across ranks.
    def _match_unused_params_allreduce(self):
        locally_used_param_map = self.reducer._get_local_used_map()
        self.process_group.allreduce(locally_used_param_map)


    def _register_buffer_comm_hook(
        self,
        state,
        hook: Callable,
        comm_hook_location=_BufferCommHookLocation.POST_FORWARD,
    ):
        r"""
        Allow custom registration of hooks that define how buffer are synchronized across ranks.

        The hook takes in an optional state and is passed in a Dict[str, Tensor]
        corresponding to buffer names and the buffers, and can run arbitrary reductions
        on buffers as opposed to DDP's default broadcast from rank 0. This is useful for
        example if a counter needs to be summed or averaged across ranks every iteration.

        Args:
            state (Any): Optional state that is passed to the hook.
            hook (Callable): Callable with the following signature:
                         ``hook(state: object, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]``
            comm_hook_location (_BufferCommHookLocation): Enum value indicating
                            where to run the hook.
                            _BufferCommHookLocation.PRE_FORWARD means that the
                            hook will run _before_ the forward pass, and
                            _BufferCommHookLocation.POST_FORWARD means that the
                            hook will run _after_ the forward pass.

            NOTE: To maximize performance, users can return a
                List[torch.futures.Future] from their hook, and DDP will
                install and await these hooks appropriately at the end of
                the backward pass. This will ensure all buffers are
                synchronized by the end of the backward pass. If this
                setting is used, it is recommended to pass
                comm_hook_location=_BufferCommHookLocation.POST_FORWARD,
                which will trigger the hook after the forward pass.
                If _BufferCommHookLocation.PRE_FORWARD is used, users must
                ensure appropriate synchronization when manipulating GPU
                buffers in the forward pass.
        """
        assert callable(hook)
        self.buffer_hook = _BufferCommHook(
            buffer_comm_hook=hook,
            buffer_comm_hook_state=state,
            buffer_comm_hook_location=comm_hook_location,
        )

    def register_comm_hook(self, state: object, hook: Callable):
        raise NotImplementedError
        self._check_comm_hook(hook)
        assert self.logger is not None
        self.logger._set_comm_hook_name(hook.__qualname__)
        self._comm_hooks.append((hook, state))
        dist._register_comm_hook(self.reducer, state, hook)

    def _register_builtin_comm_hook(self, comm_hook_type):
        r"""
        Register a built-in communication hook that specifies how DDP aggregates gradients across multiple workers.

        The built-in hooks aim to provide efficient C++ implementations for certain hooks,
        which might not be as efficient if implemented in Python using a Python communication hook.

        Args:
            comm_hook_type (dist.BuiltinCommHookType): type of communication hook, such as ALLREDUCE, FP16_COMPRESS, etc.

        .. warning ::
            DDP communication hook can only be registered once and should be registered
            before calling backward.

        Example::
            Below is an example of a FP16 compression where gradients are
            compressed into 16-bit floating-point numbers before allreduce, and
            then decompressed after allreduce.

            >>> # xdoctest: +SKIP('undefined name')
            >>> ddp._register_builtin_comm_hook(dist.BuiltinCommHookType.FP16_COMPRESS)

        """
        assert self.logger is not None
        self.logger._set_comm_hook_name(str(comm_hook_type))
        dist._register_builtin_comm_hook(self.reducer, comm_hook_type)

    def _register_fused_optim(self, optim: type, *args, optim_params=None, **kwargs):
        r"""
        Register an optimizer in DDP to optimize parameter immediately after its gradient reduction.

        Registers an optimizer with DDP such that the optimization for a
        parameter will run immediately when that parameter's gradient is
        finished with reduction, instead of waiting for all parameters'
        gradients to finish reduction. This can result in a training speedup
        depending on your workload since the optimizer can run while gradient
        reduction for other parameters are still ongoing. In addition, this has
        the potential to reduce peak memory consumption during training, as it
        only needs to load the per-parameter optimizer states of a single
        parameter at a time, instead of loading all per-parameter optimizer
        states at once.

        Args:
            optim (Type): a ``torch.optim.Optimizer`` class to be registered
            as a fused optimizer.
            *args (Sequence[Any]): Arguments to forward to `optim`.
            optim_params (Optional[Iterable[torch.Tensor]]): Set of parameters
            to optimize, similar to `params` argument of traditional `torch.optim`
            Optimizers. If this is omitted, all DDP model parameters will be
            optimized.
            **kwargs: (Dict[str, Any]): Keyword arguments to forward to `optim`.

        .. warning ::
            _register_fused_optim should only be called once on a DDP instance,
            and registering multiple fused optimizers for the same DDP model
            is not currently supported. Please ping
            https://github.com/pytorch/pytorch/issues/71595 if this is necessary
            for your use case.

        .. warning ::
            _register_fused_optim and register_comm_hook currently do not
            compose together, meaning that custom DDP communication hooks are
            not supported with overlapped optimizers. Please ping
            https://github.com/pytorch/pytorch/issues/71595 if this is necessary
            for your use case.

        .. warning ::
            Gradient accumulation and DDP `no_sync` are currently not supported
            with overlapped optimizer. Please ping
            https://github.com/pytorch/pytorch/issues/71595 if this is necessary
            for your use case.

        Example::

            >>> # xdoctest: +SKIP("No rendezvous handler")
            >>> torch.distributed.init_process_group(backend='nccl', world_size=4, init_method='...')
            >>> net = torch.nn.parallel.DistributedDataParallel(model, pg)
            >>> lr = 1e-2
            >>> betas = (0.9, 0.99)
            >>> eps = 1e-6
            >>> net._register_fused_optim(torch.optim.Adam, lr, betas=betas, eps=eps)
            >>> # Example with subset of parameters
            >>> params_to_opt = [list(net.parameters())[0]]
            >>> net._register_fused_optim(
            ...   torch.optim.Adam, lr, optim_params=params_to_opt,  betas=betas, eps=eps
            ... )
        """
        # Note: importing in function, otherwise this will cause a circular
        # import as optimizer_overlap module needs to import DistributedDataParallel.
        from torch.distributed.algorithms._optimizer_overlap import _as_overlapped_optim

        overlapped_optim = _as_overlapped_optim(optim, optim_params, *args, **kwargs)
        try:
            overlapped_optim.register_ddp(self)
        except NotImplementedError as e:
            raise RuntimeError(
                f"{optim} does not support overlapped DDP. Please file an issue to PyTorch or the respective owner of {optim}."
            ) from e

    def _distributed_broadcast_coalesced(
        self, tensors, buffer_size, authoritative_rank=0
    ):
        dist._broadcast_coalesced(
            self.process_group, tensors, buffer_size, authoritative_rank
        )

    def _check_sync_bufs_post_fwd(self):
        return (
            self.will_sync_module_buffers()
            and hasattr(self, "buffer_hook")
            and self.buffer_hook.buffer_comm_hook_location
            == _BufferCommHookLocation.POST_FORWARD
        )

    def _check_sync_bufs_pre_fwd(self):
        return self.will_sync_module_buffers() and (
            not hasattr(self, "buffer_hook")
            or self.buffer_hook.buffer_comm_hook_location
            == _BufferCommHookLocation.PRE_FORWARD
        )

    def will_sync_module_buffers(self):
        return (
            self.require_forward_param_sync
            and self.broadcast_buffers
            and len(self.modules_buffers) > 0
        )

    def _find_common_rank(self, input_rank, rank_cond):
        # -1 indicates that this rank is not under consideration to be the
        # common_rank
        rank_to_use = torch.tensor(
            [input_rank if rank_cond else -1],
            device=self.device,
        )
        dist.all_reduce(rank_to_use, op=ReduceOp.MAX, group=self.process_group)
        if rank_to_use.item() == -1:
            self._log_and_throw(
                ValueError,
                "BUG! Expected rank_cond to be true for at least one process."
                " This indicates a bug in PyTorch, please report an issue.",
            )
        return rank_to_use.item()

    def _sync_buffers(self):
        with torch.no_grad():
            # module buffer sync
            # Synchronize buffers across processes.
            # If we are running DDP with the join manager, we have to agree
            # upon a rank to sync module buffers from, since rank 0 may
            # already have been joined and have stale module buffers.
            if self._join_config.enable:
                authoritative_rank = self._find_common_rank(
                    self._distributed_rank, True
                )
            else:
                # The process with rank 0 is considered the authoritative copy.
                authoritative_rank = 0
            # Update self.modules_buffers in case any buffers were
            # reassigned.
            self._assign_modules_buffers()
            self._sync_module_buffers(authoritative_rank)

    def _sync_module_buffers(self, authoritative_rank):
        if not hasattr(self, "buffer_hook"):
            self._default_broadcast_coalesced(authoritative_rank=authoritative_rank)
        else:
            hook = self.buffer_hook.buffer_comm_hook
            state = self.buffer_hook.buffer_comm_hook_state
            futs = hook(state, self.named_module_buffers)
            if futs is not None:
                self.reducer._install_post_backward_futures(futs)

    def _default_broadcast_coalesced(
        self, bufs=None, bucket_size=None, authoritative_rank=0
    ):
        """
        Broadcasts buffers from rank 0 to rest of workers.

        If bufs, bucket_size are None, default values self.modules_buffers
        and self.broadcast_bucket_size are used instead.
        """
        if bufs is None:
            bufs = self.modules_buffers
        if bucket_size is None:
            bucket_size = self.broadcast_bucket_size

        self._distributed_broadcast_coalesced(bufs, bucket_size, authoritative_rank)

    def _passing_sync_batchnorm_handle(self, module):
        for layer in module.modules():
            if isinstance(layer, torch.nn.modules.SyncBatchNorm):
                if self.device_type == "cpu":
                    self._log_and_throw(
                        ValueError,
                        "SyncBatchNorm layers only work with GPU modules",
                    )

    def _check_comm_hook(self, hook):
        if not callable(hook):
            self._log_and_throw(TypeError, "Communication hook must be callable.")

        sig = inspect.signature(hook)
        if (
            sig.parameters["bucket"].annotation != inspect._empty
            and sig.parameters["bucket"].annotation != dist.GradBucket
        ):
            self._log_and_throw(
                ValueError,
                "Communication hook: bucket annotation should be dist.GradBucket.",
            )

        if (
            sig.return_annotation != inspect._empty
            and sig.return_annotation != torch.futures.Future[torch.Tensor]
        ):
            self._log_and_throw(
                ValueError,
                "Communication hook: return annotation should be torch.futures.Future[torch.Tensor].",
            )

        if hook.__name__ in ["bf16_compress_hook", "bf16_compress_wrapper_hook"]:
            cuda_supported = (
                torch.version.cuda is not None
            ) or torch.version.hip is not None
            nccl_supported = (
                dist.is_available()
                and dist.is_nccl_available()
                and torch.cuda.nccl.version() >= (2, 10)
            )
            xpu_xccl_supported = (
                dist.is_available()
                and dist.is_xccl_available()
                and torch.xpu.is_available()
            )

            if not ((cuda_supported and nccl_supported) or xpu_xccl_supported):
                self._log_and_throw(
                    TypeError,
                    "BF16 all reduce communication hook required CUDA 11+ and NCCL 2.10+ or XPU and XCCL",
                )

    @property
    def _distributed_rank(self):
        return dist.get_rank(self.process_group)



    @staticmethod
    def _set_params_and_buffers_to_ignore_for_model(
        module, params_and_buffers_to_ignore
    ):
        """
        Set parameters and buffers to be ignored by DDP.

        Expected format for parameters is the fully qualified name: {module_name}.{param_name}, and
        similarly, {module_name}.{buffer_name} for buffers. For example:
        params_to_ignore = []
        # NB: model here is vanilla PyTorch module, not yet wrapped with DDP.
        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if should_ignore(param):
                    # Create expected format
                    fqn = f"{module_name}.{param_name}"
                    params_to_ignore.append(fqn)
        torch.nn.parallel.DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(
            model,
            params_to_ignore
        )
        """
        # This is a workaround to set parameters and buffers DDP should ignore
        # during synchronization. It will be removed when the API is finalized
        # as part of addressing https://github.com/pytorch/pytorch/issues/43690.
        module._ddp_params_and_buffers_to_ignore = params_and_buffers_to_ignore
        for name, param in module.named_parameters():
            if name in params_and_buffers_to_ignore:
                param._ddp_ignored = True
        for name, buffer in module.named_buffers():
            if name in params_and_buffers_to_ignore:
                buffer._ddp_ignored = True


    def _set_ddp_sink_clone(self, val: bool):
        """
        Sets whether or not DDPSink should clone the output tensors or not.
        The default is True since if the loss is modified in place we run
        into the view is modified in-place error.

        Although, cloning the tensors can add significant memory and
        performance hit if the number and size of tensors are large. As
        a result, this can be set to False if you are not modifying the
        loss in place.
        """
        self._ddp_sink_clone = val

def _fp32_post_grad_acc_hook(param: torch.Tensor):
    g = param.grad
    if g is None:
        return
    # upcast and accumulate in-place (no graph)

    if param._main_grad_fp32 is None: # type: ignore[attr-defined]
        # first time init
        param._main_grad_fp32 = g.to(torch.float32) # type: ignore[attr-defined]
    else:
        # param._main_grad_fp32.add_(g.to(torch.float32)) # type: ignore[attr-defined]
        param._main_grad_fp32.add_(g) # type: ignore[attr-defined]
    # drop BF16 .grad to avoid double-accum & save memory
    param.grad = None
    # print(f'rank {dist.get_rank()} shape={param.shape} param._main_grad_fp32={param._main_grad_fp32}') # type: ignore[attr-defined]

