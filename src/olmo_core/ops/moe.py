from typing import Any, List, Optional, Tuple, Union

import torch
import torch.distributed as dist

try:
    from olmo_core.kernels import moe as kernels
except (ImportError, RuntimeError):
    kernels = None  # type: ignore


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


class GatherOp(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx: Any,
        x: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        bins: torch.Tensor,
        top_k: int,
    ):
        assert kernels is not None
        ctx.save_for_backward(indices, bin_ids, bins)
        ctx.top_k = top_k
        return kernels.gather(x, indices, bin_ids, None, bins, top_k)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx: Any, grad: torch.Tensor):
        assert kernels is not None
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
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx: Any,
        x: torch.Tensor,
        indices: torch.Tensor,
        bin_ids: torch.Tensor,
        weights: Optional[torch.Tensor],
        bins: torch.Tensor,
        top_k: int,
    ) -> torch.Tensor:
        assert kernels is not None
        maybe_x = [x] if ctx.needs_input_grad[3] else []
        ctx.save_for_backward(indices, bin_ids, weights, bins, *maybe_x)
        ctx.top_k = top_k
        ctx.x_shape = x.shape
        return kernels.scatter(x, indices, bin_ids, weights, bins, top_k)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx: Any, grad: torch.Tensor):
        assert kernels is not None

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


class BinnedGatherOp(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx: Any,
        x: torch.Tensor,
        indices: torch.Tensor,
        bins: torch.Tensor,
        bin_size: int,
        top_k: int,
    ):
        assert kernels is not None
        ctx.save_for_backward(indices, bins)
        ctx.top_k = top_k
        return kernels.binned_gather(x, indices, None, bins, bin_size, top_k)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx: Any, grad: torch.Tensor):
        assert kernels is not None
        grad = grad.contiguous()
        indices, bins = ctx.saved_tensors
        out = kernels.binned_scatter(grad, indices, None, bins, ctx.top_k)
        return out, None, None, None, None


def binned_gather(
    x: torch.Tensor, indices: torch.Tensor, bins: torch.Tensor, bin_size: int, top_k: int
) -> torch.Tensor:
    return BinnedGatherOp.apply(x, indices, bins, bin_size, top_k)  # type: ignore


class BinnedScatterOp(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx: Any,
        x: torch.Tensor,
        indices: torch.Tensor,
        weights: Optional[torch.Tensor],
        bins: torch.Tensor,
        top_k: int,
    ):
        assert kernels is not None

        assert len(x.size()) == 3
        ctx.bin_size = x.size(1)
        ctx.top_k = top_k

        # TODO: Don't save 'x' for backwards if we don't need to
        # calculate the gradient w.r.t. 'weights'.
        ctx.save_for_backward(x, indices, weights, bins)
        return kernels.binned_scatter(x, indices, weights, bins, top_k)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx: Any, grad: torch.Tensor):
        assert kernels is not None

        grad = grad.contiguous()
        x, indices, weights, bins = ctx.saved_tensors
        out = kernels.binned_gather(
            grad,
            indices,
            weights,
            bins,
            ctx.bin_size,
            ctx.top_k,
        )

        wgrad = None
        if ctx.needs_input_grad[2]:
            wgrad = kernels.binned_scatter_wgrad(
                x,
                grad,
                indices,
                bins,
                ctx.top_k,
            )
        return out, None, wgrad, None, None


def binned_scatter(
    x: torch.Tensor,
    indices: torch.Tensor,
    weights: Optional[torch.Tensor],
    bins: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    return BinnedScatterOp.apply(x, indices, weights, bins, top_k)  # type: ignore


def repeat(x: torch.Tensor, tiling: Union[torch.Size, Tuple[int, ...]]) -> torch.Tensor:
    if all((t == 1 for t in tiling)):
        return x
    return x.repeat(*tiling)


class AllToAllOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, output_split_sizes, input_split_sizes, group, async_op):
        if output_split_sizes is not None:
            out = torch.empty(
                (sum(output_split_sizes),) + x.shape[1:], device=x.device, dtype=x.dtype
            )
        else:
            out = torch.empty_like(x)

        ctx.input_shape = x.shape
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        ctx.group = group
        handle = dist.all_to_all_single(
            out,
            x.contiguous(),
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
                grad.contiguous(),
                output_split_sizes=ctx.input_split_sizes,
                input_split_sizes=ctx.output_split_sizes,
                group=ctx.group,
            )
            return out, None, None, None, None
        return None, None, None, None, None


def all_to_all(
    x: torch.Tensor,
    output_split_sizes: Optional[List[int]] = None,
    input_split_sizes: Optional[List[int]] = None,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
) -> Tuple[torch.Tensor, dist.Work]:
    return AllToAllOp.apply(  # type: ignore
        x,
        output_split_sizes,
        input_split_sizes,
        group,
        async_op,
    )



class AllToAllAsncOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, output_split_sizes, input_split_sizes, group):
        if output_split_sizes is not None:
            out = torch.empty(
                (sum(output_split_sizes),) + x.shape[1:], device=x.device, dtype=x.dtype
            )
        else:
            out = torch.empty_like(x)

        ctx.input_shape = x.shape
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        ctx.group = group
        handle = dist.all_to_all_single(
            out,
            x.contiguous(),
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
            async_op=True,
        )
        # Set the attributes on the output tensor, which will be carried to AllToAllWaitOp
        setattr(out, "_a2a_output_split_sizes", output_split_sizes)
        setattr(out, "_a2a_input_split_sizes", input_split_sizes)
        setattr(out, "_a2a_group", group)

        return out, handle

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad = grad_outputs[0]
        grad_handle = grad._a2a_handle
        grad_handle.wait()
        del grad._a2a_handle
        return grad, None, None, None, None


class AllToAllWaitOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, x_handle):
        ctx.input_shape = x.shape
        ctx.output_split_sizes = x._a2a_output_split_sizes
        ctx.input_split_sizes = x._a2a_input_split_sizes
        ctx.group = x._a2a_group
        # remove attributes
        del x._a2a_output_split_sizes
        del x._a2a_input_split_sizes
        del x._a2a_group
        
        x_handle.wait()
        return x

    @staticmethod
    def backward(ctx, *grad_outputs):

        grad = grad_outputs[0]
        out = torch.empty(
            ctx.input_shape,
            device=grad.device,
            dtype=grad.dtype,
        )
        handle = dist.all_to_all_single(
            out,
            grad.contiguous(),
            output_split_sizes=ctx.input_split_sizes,
            input_split_sizes=ctx.output_split_sizes,
            group=ctx.group,
            async_op=True
        )
        setattr(out, "_a2a_handle", handle)
        return out, None, None, None, None


def all_to_all_async(
    x: torch.Tensor,
    output_split_sizes: Optional[List[int]] = None,
    input_split_sizes: Optional[List[int]] = None,
    group: Optional[dist.ProcessGroup] = None,
) -> Tuple[torch.Tensor, dist.Work]:
    return AllToAllAsncOp.apply(  # type: ignore
        x,
        output_split_sizes,
        input_split_sizes,
        group,
    )

def all_to_all_wait(
    x: torch.Tensor,
    x_handle: dist.Work
) -> torch.Tensor:
    return AllToAllWaitOp.apply(  # type: ignore
        x,
        x_handle
    )


def _example():
    x = torch.randn(8, 16).cuda()
    x.requires_grad = True

    y0, y_handle = all_to_all_async(x)
    z = x * 2 # some random compute
    y = all_to_all_wait(y0, y_handle)
    out = y + z
    
    # backward order:
    # backward(out)
    # launch backward(all_to_all_async)
    # backward(compute)
    # wait all2all
    # backward(x)

def sum_tensor(x: torch.Tensor, dim: int = 0) -> torch.Tensor:
    if x.shape[dim] == 1:
        return x.squeeze(dim=dim)
    return x.sum(dim=dim)


def batched_histc(x: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    A batched version of ``torch.histc``.
    """
    hist = torch.zeros((*x.shape[:-1], num_classes), dtype=x.dtype, device=x.device)
    ones = torch.ones_like(x)
    hist.scatter_add_(-1, x, ones)
    return hist


def histc(x: torch.Tensor, num_classes: int) -> torch.Tensor:
    # NOTE: 'torch.histc' not implemented for integers on CPU, so convert to float then back to ints on CPU.
    if x.device.type == "cpu":
        return torch.histc(x.float(), bins=num_classes, min=0, max=num_classes - 1).int()
    else:
        return torch.histc(x, bins=num_classes, min=0, max=num_classes - 1)
