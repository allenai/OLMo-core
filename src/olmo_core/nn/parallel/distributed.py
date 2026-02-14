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
else:
    _get_default_group = None

if TYPE_CHECKING:
    from torch.utils.hooks import RemovableHandle


__all__ = ["MultiGroupDistributedDataParallel"]

logger = logging.getLogger(__name__)



@dataclass
class _GradBucket:
    process_group: Any
    storage_dtype: torch.dtype
    comm_dtype: torch.dtype
    params: list[torch.nn.Parameter]
    ranges: list[tuple[int, int]]
    numel: int
    flat_storage: torch.Tensor
    flat_comm: Optional[torch.Tensor]


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
            if _get_default_group is None:
                self.process_group = None
            else:
                self.process_group = _get_default_group()
        else:
            self.process_group = process_group

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
            raise RuntimeError(
                "DistributedDataParallel is not needed when a module doesn't have any parameter that requires a gradient.",
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
            param_process_group_fn = (
                lambda _name, _param: self.process_group
            )  # default to single process group
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

        self._grad_buckets: list[_GradBucket] = []
        self._param_to_bucket_idx: Dict[torch.nn.Parameter, int] = {}
        self._param_to_bucket_view: Dict[torch.nn.Parameter, torch.Tensor] = {}
        self._bucket_ready_count: list[int] = []
        self._grad_reduce_hooks: list[tuple[Any, int]] = []
        self._grad_views_need_rebind = False
        self._warned_grad_view_rebind = False
        self._forwards_since_finalize = 0

        self._build_grad_buckets()
        self._bind_bucket_views(zero_buffers=True, reason="initialization")

        self._fp32_acc_hooks = []
        if self._accumulate_grads_in_fp32:
            for p in module.parameters():
                if not p.requires_grad:
                    continue
                self._fp32_acc_hooks.append(
                    p.register_post_accumulate_grad_hook(self._fp32_post_grad_acc_hook)
                )

        # Register the AccumulateGrad post hooks if optimize_ddp is
        # True. The hooks will be deregistered if compiled_autograd is not
        # enabled.
        self._accum_grad_hooks: list[RemovableHandle] = []

        self._param_grad_ready: OrderedDict[torch.nn.Parameter, bool] = OrderedDict()
        self._next_reduce_bucket_idx = 0

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

    
    def _get_storage_dtype_for_param(self, param: torch.nn.Parameter) -> torch.dtype:
        if self._accumulate_grads_in_fp32:
            return torch.float32
        return param.dtype

    def _get_comm_dtype_for_storage(self, storage_dtype: torch.dtype) -> torch.dtype:
        if self._reduce_grads_in_fp32:
            return torch.float32
        return storage_dtype

    def _get_param_grad_buffer(self, param: torch.nn.Parameter) -> Optional[torch.Tensor]:
        if self._accumulate_grads_in_fp32:
            return getattr(param, "_main_grad_fp32", None)
        return param.grad

    def _set_param_grad_buffer(self, param: torch.nn.Parameter, buffer: Optional[torch.Tensor]) -> None:
        if self._accumulate_grads_in_fp32:
            param._main_grad_fp32 = buffer  # type: ignore[attr-defined]
        else:
            param.grad = buffer

    def _is_expected_grad_view(self, current: torch.Tensor, expected: torch.Tensor) -> bool:
        return (
            current.dtype == expected.dtype
            and current.device == expected.device
            and current.shape == expected.shape
            and current.stride() == expected.stride()
            and current.data_ptr() == expected.data_ptr()
        )

    def _build_grad_buckets(self) -> None:
        params_in_reduce_order = [p for p in self._reversed_module_parameters if p.requires_grad]

        current_params: list[torch.nn.Parameter] = []
        current_ranges: list[tuple[int, int]] = []
        current_numel = 0
        current_bucket_bytes = 0
        current_process_group = None
        current_storage_dtype: Optional[torch.dtype] = None
        current_comm_dtype: Optional[torch.dtype] = None

        def flush_current_bucket() -> None:
            nonlocal current_params
            nonlocal current_ranges
            nonlocal current_numel
            nonlocal current_bucket_bytes
            nonlocal current_process_group
            nonlocal current_storage_dtype
            nonlocal current_comm_dtype

            if not current_params:
                return

            assert current_storage_dtype is not None
            assert current_comm_dtype is not None
            assert current_process_group is not None

            flat_storage = torch.zeros(
                current_numel, device=self.device, dtype=current_storage_dtype
            )
            flat_comm = None
            if current_comm_dtype != current_storage_dtype:
                flat_comm = torch.empty(current_numel, device=self.device, dtype=current_comm_dtype)

            bucket_idx = len(self._grad_buckets)
            self._grad_buckets.append(
                _GradBucket(
                    process_group=current_process_group,
                    storage_dtype=current_storage_dtype,
                    comm_dtype=current_comm_dtype,
                    params=list(current_params),
                    ranges=list(current_ranges),
                    numel=current_numel,
                    flat_storage=flat_storage,
                    flat_comm=flat_comm,
                )
            )

            for param, (start, end) in zip(current_params, current_ranges):
                self._param_to_bucket_idx[param] = bucket_idx
                self._param_to_bucket_view[param] = flat_storage[start:end].view_as(param)

            current_params = []
            current_ranges = []
            current_numel = 0
            current_bucket_bytes = 0
            current_process_group = None
            current_storage_dtype = None
            current_comm_dtype = None

        for param in params_in_reduce_order:
            process_group = self.param_to_process_group[param]
            storage_dtype = self._get_storage_dtype_for_param(param)
            comm_dtype = self._get_comm_dtype_for_storage(storage_dtype)
            param_bytes = param.numel() * torch.empty((), dtype=comm_dtype).element_size()

            should_flush = (
                current_params
                and (
                    process_group is not current_process_group
                    or storage_dtype != current_storage_dtype
                    or comm_dtype != current_comm_dtype
                    or (current_bucket_bytes + param_bytes > self.bucket_bytes_cap)
                )
            )
            if should_flush:
                flush_current_bucket()

            start = current_numel
            end = start + param.numel()
            current_params.append(param)
            current_ranges.append((start, end))
            current_numel = end
            current_bucket_bytes += param_bytes
            current_process_group = process_group
            current_storage_dtype = storage_dtype
            current_comm_dtype = comm_dtype

        flush_current_bucket()
        self._bucket_ready_count = [0 for _ in self._grad_buckets]

    def _bind_bucket_views(self, *, zero_buffers: bool, reason: str) -> None:
        if zero_buffers:
            for bucket in self._grad_buckets:
                bucket.flat_storage.zero_()

        rebound_count = 0
        none_count = 0
        for param, expected_view in self._param_to_bucket_view.items():
            current = self._get_param_grad_buffer(param)
            if current is None:
                none_count += 1
            elif not self._is_expected_grad_view(current, expected_view):
                raise RuntimeError(
                    "Detected an external gradient tensor replacement that breaks bucket views. "
                    "This mode requires grads to remain bucket views."
                )

            if current is not expected_view:
                self._set_param_grad_buffer(param, expected_view)
                rebound_count += 1

            # In fp32-accum mode, .grad must stay ephemeral and be consumed by the post-acc hook.
            if self._accumulate_grads_in_fp32:
                param.grad = None

        self._grad_views_need_rebind = False
        if rebound_count > 0 and reason != "initialization":
            if not self._warned_grad_view_rebind:
                logger.warning(
                    f"Rebound {rebound_count} gradient bucket view(s) because buffers were detached "
                    f"(reason={reason}, none={none_count})."
                )
                self._warned_grad_view_rebind = True

    def _ensure_grad_views_bound(self, *, allow_none_rebind: bool, where: str) -> None:
        if self._grad_views_need_rebind:
            if self._forwards_since_finalize > 0:
                raise RuntimeError(
                    f"Gradient buckets were marked for rebind in {where} after the training step started. "
                    "This usually indicates set_to_none=True between micro-batches."
                )
            self._bind_bucket_views(zero_buffers=True, reason=f"flagged@{where}")
            return

        has_none_binding = False
        for param, expected_view in self._param_to_bucket_view.items():
            current = self._get_param_grad_buffer(param)
            if current is None:
                has_none_binding = True
                continue
            if not self._is_expected_grad_view(current, expected_view):
                raise RuntimeError(
                    f"Gradient view integrity check failed in {where}: gradient tensor was replaced "
                    "with a non-bucket-view tensor."
                )

        if has_none_binding:
            if not allow_none_rebind:
                raise RuntimeError(
                    f"Found None gradient buffers in {where}. This usually means set_to_none=True ran "
                    "at an unexpected time. Rebind before backward by running a forward pass."
                )
            if self._forwards_since_finalize > 0 and not self._grad_views_need_rebind:
                raise RuntimeError(
                    f"Found detached gradient buffers in {where} after this training step already started. "
                    "Rebinding now would lose accumulated grads. This usually indicates an unexpected "
                    "set_to_none=True between micro-batches."
                )
            self._bind_bucket_views(zero_buffers=True, reason=f"none@{where}")

    def _fp32_post_grad_acc_hook(self, param: torch.Tensor):
        g = param.grad
        if g is None:
            return

        expected_view = self._param_to_bucket_view[param]
        main_grad = getattr(param, "_main_grad_fp32", None)
        if main_grad is None:
            raise RuntimeError(
                "FP32 grad bucket view is missing during backward. "
                "Likely caused by set_to_none=True after forward began."
            )
        if not self._is_expected_grad_view(main_grad, expected_view):
            raise RuntimeError(
                "FP32 grad buffer is not the expected bucket view. "
                "External grad buffer replacement is not supported in bucket-view mode."
            )

        main_grad.add_(g)
        param.grad = None

    def _launch_bucket_all_reduce(self, bucket_idx: int) -> None:
        if self._comm_hooks:
            raise NotImplementedError("Comm hooks are not implemented in bucket-view mode.")

        bucket = self._grad_buckets[bucket_idx]
        world_size = bucket.process_group.size()

        if bucket.storage_dtype == bucket.comm_dtype:
            tensor_for_reduce = bucket.flat_storage
            tensor_for_reduce.div_(world_size)
        else:
            assert bucket.flat_comm is not None
            bucket.flat_comm.copy_(bucket.flat_storage)
            bucket.flat_comm.div_(world_size)
            tensor_for_reduce = bucket.flat_comm

        handle = torch.distributed.all_reduce(
            tensor_for_reduce, op=ReduceOp.SUM, group=bucket.process_group, async_op=True
        )
        self._grad_reduce_hooks.append((handle, bucket_idx))

    def _maybe_kick_start_all_reduce(self):
        while self._next_reduce_bucket_idx < len(self._grad_buckets):
            bucket = self._grad_buckets[self._next_reduce_bucket_idx]
            if self._bucket_ready_count[self._next_reduce_bucket_idx] < len(bucket.params):
                break

            self._launch_bucket_all_reduce(self._next_reduce_bucket_idx)
            self._next_reduce_bucket_idx += 1


    def _register_accum_grad_hook(self):
        def notify_grad_ready(
            param,
        ):
            if not self.require_backward_grad_sync:
                return

            if self._param_grad_ready[param]:
                return

            self._param_grad_ready[param] = True
            bucket_idx = self._param_to_bucket_idx[param]
            self._bucket_ready_count[bucket_idx] += 1

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
        # Grad buffers should already be bound before backward starts. If they are detached here,
        # we cannot safely rebind without risking silent corruption.
        self._ensure_grad_views_bound(allow_none_rebind=False, where="finalize_grad_reduce")

        # in some cases (eg, imbalance moe routing), some params may not have grads, and their
        # post_accumulate_grad_hook is never called, so their grad_ready is never set to True.
        if self._next_reduce_bucket_idx < len(self._grad_buckets):
            for param in self._param_grad_ready.keys():
                if not self._param_grad_ready[param]:
                    self._param_grad_ready[param] = True
                    bucket_idx = self._param_to_bucket_idx[param]
                    self._bucket_ready_count[bucket_idx] += 1

                    # Keep missing grads explicitly zero in the bucket view.
                    self._param_to_bucket_view[param].zero_()

            self._maybe_kick_start_all_reduce()

        # now all grad reduce should have been launched
        assert self._next_reduce_bucket_idx == len(self._grad_buckets), (
            f"Not all bucket all-reduce operations were launched: "
            f"{self._next_reduce_bucket_idx} vs {len(self._grad_buckets)}"
        )


        for idx, (handle, bucket_idx) in enumerate(self._grad_reduce_hooks):
            handle.wait()
            bucket = self._grad_buckets[bucket_idx]
            if bucket.flat_comm is not None:
                bucket.flat_storage.copy_(bucket.flat_comm)
        self._grad_reduce_hooks = []

        self._next_reduce_bucket_idx = 0
        
        # mark all grads as not ready
        for key in self._param_grad_ready.keys():
            self._param_grad_ready[key] = False
        for bucket_idx in range(len(self._bucket_ready_count)):
            self._bucket_ready_count[bucket_idx] = 0
        self._forwards_since_finalize = 0


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
        self._ensure_grad_views_bound(allow_none_rebind=True, where="forward")
        self._forwards_since_finalize += 1
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

    def zero_grad(self, set_to_none: bool = True):
        super().zero_grad(set_to_none=set_to_none)

        if self._accumulate_grads_in_fp32:
            if set_to_none:
                for param in self._module_parameters:
                    if not param.requires_grad:
                        continue
                    param._main_grad_fp32 = None  # type: ignore[attr-defined]
            else:
                for param in self._module_parameters:
                    if not param.requires_grad:
                        continue
                    view = self._param_to_bucket_view[param]
                    view.zero_()
                    param._main_grad_fp32 = view  # type: ignore[attr-defined]

        self._forwards_since_finalize = 0
        if set_to_none:
            self._grad_views_need_rebind = True

    def set_main_grads_to_none(self):
        if hasattr(self.module, "set_main_grads_to_none"):
            self.module.set_main_grads_to_none()
        else:
            for param in self._module_parameters:
                if not param.requires_grad:
                    continue
                if hasattr(param, "_main_grad_fp32"):
                    param._main_grad_fp32 = None  # type: ignore[attr-defined]
        self._grad_views_need_rebind = True
        self._forwards_since_finalize = 0


    def register_comm_hook(self, state: object, hook: Callable):
        raise NotImplementedError

    @property
    def _distributed_rank(self):
        return dist.get_rank(self.process_group)
