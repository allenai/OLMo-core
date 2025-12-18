# mypy: allow-untyped-defs

import logging
import os
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
from torch.nn.modules import Module

from collections import OrderedDict

if dist.is_available():
    from torch.distributed.distributed_c10d import (
        _get_default_group,
        _rank_not_in_group,
        ReduceOp,
    )
    from torch.distributed.utils import (
        _verify_param_shape_across_processes,
    )

if TYPE_CHECKING:
    from torch.utils.hooks import RemovableHandle


__all__ = ["MultiGroupDistributedDataParallel"]

logger = logging.getLogger(__name__)



class MultiGroupDistributedDataParallel(Module):
    

    def __init__(
        self,
        module,
        # device_ids=None,
        # output_device=None,
        dim=0,
        # broadcast_buffers=True,
        init_sync=True,
        process_group=None,
        bucket_cap_mb=None,
        ####### 
        param_process_group_fn=None,  # NEW
        accumulate_grads_in_fp32=False,  # NEW
        reduce_grads_in_fp32=False,  # NEW
    ):
        super().__init__()
        # Joinable.__init__(self)
        _use_python_reducer = (
            torch._dynamo.utils.get_optimize_ddp_mode() == "python_reducer"
        )
        if not _use_python_reducer:
             assert False, "Only python_reducer is supported. Please set torch._dynamo.config.optimize_ddp = \"python_reducer\""

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

        self.dim = dim
        self.module = module
        self.device = next(iter(self._module_parameters)).device
        # self.broadcast_buffers = broadcast_buffers
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

        # the hook that controls gradient allreduce
        self._register_accum_grad_hook()



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
            assert False, "Not implemented comm hooks with multi-process-group DDP"
            for hook, state in self._comm_hooks:
                hook(state, (param.grad, param))
        else:
            if self._accumulate_grads_in_fp32:
                # use _main_grad_fp32
                assert self._reduce_grads_in_fp32, "reduce_grads_in_fp32 must be True when accumulate_grads_in_fp32 is True"

                # gradient = param._main_grad_fp32 / process_group.size()
                param._main_grad_fp32.div_(process_group.size())

                handle = torch.distributed.all_reduce(param._main_grad_fp32, op=ReduceOp.SUM, group=process_group, async_op=True)
                self._grad_reduce_hooks.append((handle, None, None)) # no need to copy

                # param._main_grad_fp32.copy_(gradient)
            else:
                # use param.grad (bfloat16)
                if self._reduce_grads_in_fp32:
                    gradient = param.grad.float() / process_group.size()

                    handle = torch.distributed.all_reduce(gradient, op=ReduceOp.SUM, group=process_group, async_op=True)
                    self._grad_reduce_hooks.append((handle, param.grad, gradient)) # need to write back to bf16 grad

                else:
                    param.grad.div_(process_group.size())

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



    def _pre_forward(self, *inputs, **kwargs):        
        return inputs, kwargs


    def _post_forward(self, output):
        return output


    def forward(self, *inputs, **kwargs):
        with torch.autograd.profiler.record_function("MultiGroupDistributedDataParallel.forward"):
            inputs, kwargs = self._pre_forward(*inputs, **kwargs)
            output = self.module(*inputs, **kwargs)
            
            return self._post_forward(output)


    def train(self, mode=True):
        super().train(mode)
        return self


    def register_comm_hook(self, state: object, hook: Callable):
        raise NotImplementedError

    @property
    def _distributed_rank(self):
        return dist.get_rank(self.process_group)


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

