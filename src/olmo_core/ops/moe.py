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


def segment_ids_from_eos(input_ids: torch.Tensor, eos_token_id: int) -> torch.Tensor:
    """
    Compute per-token document ids on-device (no host sync), replacing the CPU
    ``torch.nonzero`` + Python-loop boundary construction used by two-level MoE routers.

    A document is the half-open span between consecutive EOS positions, with the EOS token
    itself belonging to the document it *starts* (the following one). This matches the old
    ``[start, end)`` boundary convention where the leading span excludes its terminating EOS.

    :param input_ids: token ids of shape ``(B, S)``.
    :param eos_token_id: the end-of-document token id.

    :returns: ``seg_id`` of shape ``(B, S)`` (long), where ``seg_id[b, s]`` is the index of the
        document that token ``s`` belongs to within sequence ``b`` (starting at 0).
    """
    eos = (input_ids == eos_token_id).to(torch.long)
    eos[:, 0] = 0  # drop position 0 to match the old `pos = pos[pos > 0]`
    return eos.cumsum(dim=1)


def doc_rank(doc_prob_per_token: torch.Tensor) -> torch.Tensor:
    """
    Per-token descending rank of each expert by its document-summed probability.

    :param doc_prob_per_token: ``(B, S, E)`` document-summed expert probabilities broadcast to
        every token (see :func:`doc_sum_scatter`).

    :returns: ``(B, S, E)`` long ranks where ``0`` is the highest-probability expert.
    """
    return doc_prob_per_token.argsort(dim=-1, descending=True).argsort(dim=-1)


def doc_sum_scatter(per_token: torch.Tensor, seg_id: torch.Tensor) -> torch.Tensor:
    """
    Sum a per-token quantity over each token's document, then broadcast the document total
    back to every token in that document. Fully vectorized (scatter_add + gather), no sync.

    The document axis is materialized at size ``S`` (a safe static upper bound: a length-``S``
    sequence has at most ``S`` documents), which keeps shapes compile-friendly.

    :param per_token: ``(B, S, E)`` per-token values (e.g. softmax expert probabilities).
    :param seg_id: ``(B, S)`` document ids from :func:`segment_ids_from_eos`.

    :returns: ``(B, S, E)`` where each token holds its document's summed value.
    """
    B, S, E = per_token.shape
    idx = seg_id.unsqueeze(-1).expand(-1, -1, E)
    doc_sums = torch.zeros(B, S, E, dtype=per_token.dtype, device=per_token.device)
    doc_sums.scatter_add_(1, idx, per_token)
    return doc_sums.gather(1, idx)


def pool_keep_mask(
    doc_prob_per_token: torch.Tensor,
    pool_per_token: torch.Tensor,
    num_forced: int = 0,
) -> torch.Tensor:
    """
    Vectorized two-level document pool selection: keep the top ``pool`` experts per document
    (by document-summed probability), always keeping the last ``num_forced`` experts.

    Reproduces the old per-document ``topk`` masking semantics: forced experts (the last
    ``num_forced`` experts) are excluded from the ranking and always kept, and only the
    ``pool - num_forced`` best of the remaining candidates are kept. A threshold ``< 0`` (i.e.
    ``pool < num_forced``) keeps no candidates, and ``pool >= E`` keeps everything.

    :param doc_prob_per_token: ``(B, S, E)`` document-summed expert probabilities per token.
    :param pool_per_token: ``(B, S)`` per-token pool size (constant within a document).
    :param num_forced: number of trailing experts always kept and excluded from ranking.

    :returns: ``(B, S, E)`` boolean mask, ``True`` where the expert is kept for that token.
    """
    if num_forced > 0:
        prob_for_rank = doc_prob_per_token.clone()
        prob_for_rank[..., -num_forced:] = float("-inf")
    else:
        prob_for_rank = doc_prob_per_token
    rank_keep = doc_rank(prob_for_rank)
    keep = rank_keep < (pool_per_token.unsqueeze(-1) - num_forced)
    if num_forced > 0:
        keep[..., -num_forced:] = True
    return keep
