import functools
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.distributed as dist


def _is_eligible(x):
    return x.is_floating_point() and x.is_cuda and (x.dtype is not torch.float64)


def _cast(x, dtype):
    if isinstance(x, torch.Tensor) and _is_eligible(x):
        return x.to(dtype)
    elif isinstance(x, dict):
        return {_cast(k, dtype): _cast(v, dtype) for k, v in x.items()}
    elif isinstance(x, (list, tuple)):
        return type(x)(map(lambda y: _cast(y, dtype), x))
    return x


def autocast_fwd(fwd):
    """
    Wrap a custom autograd forward function to ensure it always uses the autocast dtype.
    """

    @functools.wraps(fwd)
    def decorate_fwd(*args, **kwargs):
        if torch.is_autocast_enabled():
            with torch.autocast(device_type="cuda", enabled=False):
                dtype = torch.get_autocast_gpu_dtype()
                return fwd(*_cast(args, dtype), **_cast(kwargs, dtype))
        return fwd(*args, **kwargs)

    return decorate_fwd


def autocast_bwd(bwd):
    """
    Wrap a custom autograd backward function to ensure it always uses the autocast dtype.
    """

    @functools.wraps(bwd)
    def decorate_bwd(*args, **kwargs):
        with torch.autocast(device_type="cuda", enabled=False):
            return bwd(*args, **kwargs)

    return decorate_bwd


class GatherOp(torch.autograd.Function):
    @staticmethod
    @autocast_fwd
    def forward(
        ctx: Any,
        x: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        bins: torch.Tensor,
        top_k: int,
    ):
        from . import kernels

        ctx.save_for_backward(indices, bin_ids, bins)
        ctx.top_k = top_k
        return kernels.gather(x, indices, bin_ids, None, bins, top_k)

    @staticmethod
    @autocast_bwd
    def backward(ctx: Any, grad: torch.Tensor):
        from . import kernels

        grad = grad.contiguous()
        indices, bin_ids, bins = ctx.saved_tensors
        out = kernels.scatter(grad, indices, bin_ids, None, bins, ctx.top_k)
        return out, None, None, None, None, None


def gather(
    x: torch.Tensor,
    indices: torch.Tensor,
    bin_ids: torch.Tensor,
    bins: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    return GatherOp.apply(x, indices, bin_ids, bins, top_k)  # type: ignore


class ScatterOp(torch.autograd.Function):
    @staticmethod
    @autocast_fwd
    def forward(
        ctx: Any,
        x: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        weights: Optional[torch.Tensor],
        bins: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        from . import kernels

        maybe_x = [x] if ctx.needs_input_grad[3] else []
        ctx.save_for_backward(indices, bin_ids, weights, bins, *maybe_x)
        ctx.top_k = top_k
        ctx.x_shape = x.shape
        return kernels.scatter(x, indices, bin_ids, weights, bins, top_k)

    @staticmethod
    @autocast_bwd
    def backward(ctx: Any, grad: torch.Tensor):
        from . import kernels

        grad = grad.contiguous()
        saved_tensors = ctx.saved_tensors

        indices, bin_ids, weights, bins = saved_tensors[:4]
        dgrad = None
        if ctx.needs_input_grad[0]:
            dgrad = kernels.gather(
                grad,
                indices,
                bin_ids,
                weights,
                bins,
                ctx.top_k,
            )

        wgrad = None
        if ctx.needs_input_grad[3]:  # need wgrad
            x = saved_tensors[-1]
            wgrad = kernels.scatter_wgrad(
                x,
                grad,
                indices,
                bin_ids,
                bins,
                ctx.top_k,
            )
        return dgrad, None, None, wgrad, None, None, None


def scatter(
    x: torch.Tensor,
    indices: torch.Tensor,
    bin_ids: torch.Tensor,
    weights: Optional[torch.Tensor],
    bins: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    return ScatterOp.apply(x, indices, bin_ids, weights, bins, top_k)  # type: ignore


def repeat(x: torch.Tensor, tiling: Union[torch.Size, Tuple[int, ...]]) -> torch.Tensor:
    if all((t == 1 for t in tiling)):
        return x
    return x.repeat(*tiling)


class AllToAllOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, output_split_sizes, input_split_sizes, group, async_op):
        out = torch.empty((sum(output_split_sizes),) + x.shape[1:], device=x.device, dtype=x.dtype)

        ctx.input_shape = x.shape
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        ctx.group = group
        handle = dist.all_to_all_single(
            out,
            x,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            async_op=async_op,
        )
        return out, handle

    @staticmethod
    def backward(ctx, grad, _):
        if ctx.needs_input_grad[0]:
            out = torch.empty(
                ctx.input_shape,
                device=grad.device,
                dtype=grad.dtype,
            )
            dist.all_to_all_single(
                out,
                grad,
                output_split_sizes=ctx.input_split_sizes,
                input_split_sizes=ctx.output_split_sizes,
                group=ctx.group,
            )
            return out, None, None, None, None
        return None, None, None, None, None


def all_to_all(
    x: torch.Tensor,
    output_split_sizes: List[int],
    input_split_sizes: List[int],
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
) -> Tuple[torch.Tensor, Any]:
    return AllToAllOp.apply(  # type: ignore
        x,
        output_split_sizes,
        input_split_sizes,
        group,
        async_op,
    )


def sum_tensor(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    if x.shape[dim] == 1:
        return x.squeeze(dim=dim)
    return x.sum(dim=dim)
