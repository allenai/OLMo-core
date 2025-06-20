# Adapted from https://github.com/databricks/megablocks/blob/main/megablocks/backend/kernels.py

from typing import Optional

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore
import nvtx


def assert_is_tensor(x, ndim):
    if x.ndim != ndim:
        raise ValueError(f"Expected {ndim}-tensor but got {x.ndim}-tensor")


def assert_is_matrix(x):
    assert_is_tensor(x, 2)


def assert_is_vector(x):
    if x.ndim != 1:
        raise ValueError(f"Expected 1-tensor but got {x.ndim}-tensor")


def assert_equal(a, b):
    if a != b:
        raise ValueError(
            f"Expected dimensions to be equal but got {a} and {b}.",
        )


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_X": 64}, num_warps=2),
        triton.Config({"BLOCK_X": 128}, num_warps=2),
        triton.Config({"BLOCK_X": 256}, num_warps=2),
        triton.Config({"BLOCK_X": 128}, num_warps=4),
        triton.Config({"BLOCK_X": 256}, num_warps=4),
    ],
    key=["NUM_COLUMNS"],
)
@triton.jit
def _padded_copy(
    a,  # (tokens, hidden_size), real.
    b,
    indices,  # (tokens * top_k), integer.
    bin_ids,  # (tokens * top_k), integer.
    weights,  # (tokens * top_k), real.
    bins,  # (num_experts), integer.
    padded_bins,  # (num_experts), integer.
    NUM_COLUMNS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_X: tl.constexpr,
    A_TO_B: tl.constexpr,
    SCALE: tl.constexpr,
):
    # Our index into array 'a'.
    index_a = tl.load(indices + tl.program_id(0))

    # One threadblock per row in 'a'. Array 'b' has greater or equal
    # number of rows since they could be padded.
    bin_idx = tl.load(bin_ids + tl.program_id(0))

    # Now we know what bin we're assigned to, but we need to know how
    # many threadblocks were assigned to earlier bins so we can offset
    # in our bin properly.
    offset_in_bin = tl.program_id(0)
    if bin_idx > 0:
        offset_in_bin -= tl.load(bins + bin_idx - 1)

    # Load the starting index of our bin in array 'b'.
    index_b = offset_in_bin
    if bin_idx > 0:
        index_b += tl.load(padded_bins + bin_idx - 1)

    # Offset the input and output pointers.
    #
    # If we're going from A to B, divide the input index to copy
    # the same input repeatedly. If we're going from B to A we
    # need to reduce the result. Using atomics is slow, so we
    # do the reduce step in a second kernel.
    offset = index_a // TOP_K if A_TO_B else index_a
    a += tl.multiple_of(offset * NUM_COLUMNS, NUM_COLUMNS)
    b += tl.multiple_of(index_b * NUM_COLUMNS, NUM_COLUMNS)
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_X), BLOCK_X)

    # Load the scale, if requested.
    scale = tl.load(weights + index_a) if SCALE else 1

    # Swap the pointers depending on the direction.
    iptr = a if A_TO_B else b
    optr = b if A_TO_B else a

    iterations = tl.cdiv(NUM_COLUMNS, BLOCK_X)
    for _ in range(iterations):
        mask = offsets < NUM_COLUMNS
        x = tl.load(iptr + offsets, mask=mask)
        x = x.to(tl.float32) * scale.to(tl.float32)  # type: ignore

        tl.store(optr + offsets, x.to(optr.dtype.element_ty), mask=mask)

        offsets += BLOCK_X

@nvtx.annotate("gather", color="blue")
def gather(
    x: torch.Tensor,
    indices: torch.Tensor,
    bin_ids: torch.Tensor,
    weights: Optional[torch.Tensor],
    bins: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    # Validate the input shapes.
    assert_is_matrix(x)
    assert_is_vector(indices)
    assert_is_vector(bin_ids)
    assert_is_vector(bins)
    assert_equal(indices.shape[0], x.shape[0] * top_k)
    assert_equal(bin_ids.shape[0], x.shape[0] * top_k)

    if weights is not None:
        assert_equal(weights.shape[0], x.shape[0] * top_k)

    # NOTE: There is no padding so the output rows equals the
    # input rows multiplied by top_k.
    output_rows = x.shape[0] * top_k
    out = torch.empty((output_rows, x.shape[1]), dtype=x.dtype, device=x.device)
    _padded_copy[(indices.shape[0],)](  # type: ignore
        x,
        out,
        indices,
        bin_ids,
        weights,
        bins,
        bins,
        NUM_COLUMNS=x.shape[1],
        A_TO_B=True,
        TOP_K=top_k,
        SCALE=weights is not None,
    )
    return out


@nvtx.annotate("padded_scatter", color="blue")
def padded_scatter(
    x: torch.Tensor,
    indices: torch.Tensor,
    bin_ids: torch.Tensor,
    weights: Optional[torch.Tensor],
    bins: torch.Tensor,
    padded_bins: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    # Validate the input shapes.
    assert_is_matrix(x)
    assert_is_vector(indices)
    assert_is_vector(bin_ids)
    assert_is_vector(bins)
    assert_is_vector(padded_bins)
    assert_equal(indices.shape[0], bin_ids.shape[0])
    assert_equal(bins.size(), padded_bins.size())

    if weights is not None:
        assert_equal(indices.shape[0], weights.shape[0])

    tokens = indices.shape[0] // top_k
    out = torch.empty((tokens, top_k, x.shape[1]), dtype=x.dtype, device=x.device)
    _padded_copy[(indices.shape[0],)](  # type: ignore
        out,
        x,
        indices,
        bin_ids,
        weights,
        bins,
        padded_bins,
        NUM_COLUMNS=x.shape[1],
        A_TO_B=False,
        TOP_K=top_k,
        SCALE=weights is not None,
    )

    # Reduce along the top-k dimension, if needed.
    return out.sum(dim=1) if top_k > 1 else out.view(tokens, x.shape[1])


def scatter(
    x: torch.Tensor,
    indices: torch.Tensor,
    bin_ids: torch.Tensor,
    weights: Optional[torch.Tensor],
    bins: torch.Tensor,
    top_k: int,
):
    return padded_scatter(x, indices, bin_ids, weights, bins, bins, top_k)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_X": 64}, num_warps=2),
        triton.Config({"BLOCK_X": 128}, num_warps=2),
        triton.Config({"BLOCK_X": 256}, num_warps=2),
        triton.Config({"BLOCK_X": 128}, num_warps=4),
        triton.Config({"BLOCK_X": 256}, num_warps=4),
    ],
    key=["NUM_COLUMNS"],
)
@triton.jit
def _padded_copy_wgrad(
    x,  #  (tokens, top_k, hidden_size), real.
    grad,  # (tokens, hidden_size), real.
    wgrad,  # (tokens, top_k), real.
    indices,  # (tokens * top_k), integer.
    bin_ids,  # (tokens * top_k), integer.
    bins,  # (num_experts), integer.
    padded_bins,  # (num_experts), integer.
    NUM_COLUMNS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_X: tl.constexpr,
):
    # Our index into 'tokens * top_k'.
    index_out = tl.load(indices + tl.program_id(0))

    # One threadblock per row in 'a'. Array 'b' has greater or equal
    # number of rows since they could be padded.
    bin_idx = tl.load(bin_ids + tl.program_id(0))

    # Now we know what bin we're assigned to, but we need to know how
    # many threadblocks were assigned to earlier bins so we can offset
    # in our bin properly.
    offset_in_bin = tl.program_id(0)
    if bin_idx > 0:
        offset_in_bin -= tl.load(bins + bin_idx - 1)

    # Load the starting index of our bin in array 'x'.
    index_x = offset_in_bin
    if bin_idx > 0:
        index_x += tl.load(padded_bins + bin_idx - 1)

    # Offset the input and output pointers.
    wgrad += index_out
    grad += tl.multiple_of((index_out // TOP_K) * NUM_COLUMNS, NUM_COLUMNS)
    x += tl.multiple_of(index_x * NUM_COLUMNS, NUM_COLUMNS)
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_X), BLOCK_X)

    acc = tl.zeros((BLOCK_X,), dtype=tl.float32)
    iterations = tl.cdiv(NUM_COLUMNS, BLOCK_X)
    for _ in range(iterations):
        mask = offsets < NUM_COLUMNS
        data = tl.load(x + offsets, mask=mask).to(tl.float32)
        scale = tl.load(grad + offsets, mask=mask).to(tl.float32)
        acc += data * scale
        offsets += BLOCK_X

    # Reduce to get the final result and store.
    out = tl.sum(acc).to(wgrad.dtype.element_ty)
    tl.store(wgrad, out)


def padded_scatter_wgrad(
    x: torch.Tensor,
    grad: torch.Tensor,
    indices: torch.Tensor,
    bin_ids: torch.Tensor,
    bins: torch.Tensor,
    padded_bins: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    # Validate the input shapes.
    assert_is_matrix(x)
    assert_is_matrix(grad)
    assert_is_vector(indices)
    assert_is_vector(bin_ids)
    assert_is_vector(bins)
    assert_is_vector(padded_bins)
    assert_equal(indices.shape[0], bin_ids.shape[0])
    assert_equal(bins.size(), padded_bins.size())

    tokens = indices.shape[0] // top_k
    out = torch.empty((tokens * top_k), dtype=x.dtype, device=x.device)
    _padded_copy_wgrad[(indices.shape[0],)](  # type: ignore
        x,
        grad,
        out,
        indices,
        bin_ids,
        bins,
        padded_bins,
        NUM_COLUMNS=x.shape[1],
        TOP_K=top_k,
    )
    return out


def scatter_wgrad(
    x: torch.Tensor,
    grad: torch.Tensor,
    indices: torch.Tensor,
    bin_ids: torch.Tensor,
    bins: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    return padded_scatter_wgrad(x, grad, indices, bin_ids, bins, bins, top_k)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_X": 64}, num_warps=2),
        triton.Config({"BLOCK_X": 128}, num_warps=2),
        triton.Config({"BLOCK_X": 256}, num_warps=2),
        triton.Config({"BLOCK_X": 128}, num_warps=4),
        triton.Config({"BLOCK_X": 256}, num_warps=4),
    ],
    key=["NUM_COLUMNS"],
)
@triton.jit
def _binned_copy(
    a,  # (tokens, hidden_size), real.
    b,  # (num_experts, expert_capacity, num_columns), real.
    expert_capacity,
    indices,  # (tokens * top_k), integer.
    weights,  # (tokens * top_k), real.
    bins,  # (num_experts), integer.
    NUM_COLUMNS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_X: tl.constexpr,
    A_TO_B: tl.constexpr,
    SCALE: tl.constexpr,
):
    # Load our indices into the output.
    expert_idx = tl.program_id(0)
    entry_idx = tl.program_id(1)

    # Calculate our offset into the output.
    index_b = expert_idx * expert_capacity + entry_idx

    # Load the index bounds for our bin and calculate
    # the number of tokens assigned to our expert.
    start = 0
    if expert_idx > 0:
        start = tl.load(bins + expert_idx - 1)
    end = tl.load(bins + expert_idx)
    num_tokens = end - start

    # Calculate our offset into the input. If we don't
    # have an input exit early.
    if entry_idx >= num_tokens:
        return
    index_a = tl.load(indices + start + entry_idx)

    # Offset the input and output pointers.
    #
    # If we're going from A to B, divide the input index to copy
    # the same input repeatedly. If we're going from B to A we
    # need to reduce the result. Using atomics is slow, so we
    # do the reduce step in a second kernel.
    offset = index_a // TOP_K if A_TO_B else index_a
    a += tl.multiple_of(offset * NUM_COLUMNS, NUM_COLUMNS)
    b += tl.multiple_of(index_b * NUM_COLUMNS, NUM_COLUMNS)
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_X), BLOCK_X)

    # Load the scale, if requested.
    scale = tl.load(weights + index_a) if SCALE else 1

    # Swap the pointers depending on the direction.
    #
    # NOTE: We need to zero the output in both directions.
    iptr = a if A_TO_B else b
    optr = b if A_TO_B else a

    iterations = tl.cdiv(NUM_COLUMNS, BLOCK_X)
    for _ in range(iterations):
        mask = offsets < NUM_COLUMNS
        x = tl.load(iptr + offsets, mask=mask)
        x = x.to(tl.float32) * scale.to(tl.float32)  # type: ignore

        tl.store(optr + offsets, x.to(optr.dtype.element_ty), mask=mask)

        offsets += BLOCK_X


@nvtx.annotate("binned_gather", color="blue")
def binned_gather(
    x: torch.Tensor,
    indices: torch.Tensor,
    weights: Optional[torch.Tensor],
    bins: torch.Tensor,
    expert_capacity: int,
    top_k: int,
) -> torch.Tensor:
    # Validate the input shapes.
    assert_is_matrix(x)
    assert_is_vector(indices)
    assert_is_vector(bins)
    assert_equal(indices.shape[0], x.shape[0] * top_k)

    if weights is not None:
        assert_equal(weights.shape[0], x.shape[0] * top_k)

    num_experts = bins.shape[0]
    out = torch.zeros((num_experts, expert_capacity, x.shape[1]), dtype=x.dtype, device=x.device)

    _binned_copy[(num_experts, expert_capacity)](  # type: ignore
        x,
        out,
        expert_capacity,
        indices,
        weights,
        bins,
        NUM_COLUMNS=x.shape[1],
        A_TO_B=True,
        TOP_K=top_k,
        SCALE=weights is not None,
    )
    return out


@nvtx.annotate("binned_scatter", color="blue")
def binned_scatter(
    x: torch.Tensor,
    indices: torch.Tensor,
    weights: Optional[torch.Tensor],
    bins: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    # Validate the input shapes.
    assert_is_tensor(x, 3)
    assert_is_vector(indices)
    assert_is_vector(bins)
    assert_equal(bins.shape[0], x.shape[0])

    if weights is not None:
        assert_equal(indices.shape[0], weights.shape[0])

    num_experts, expert_capacity, hidden_size = x.shape
    tokens = indices.shape[0] // top_k
    out = torch.zeros((tokens, top_k, hidden_size), dtype=x.dtype, device=x.device)
    _binned_copy[(num_experts, expert_capacity)](  # type: ignore
        out,
        x,
        expert_capacity,
        indices,
        weights,
        bins,
        NUM_COLUMNS=hidden_size,
        A_TO_B=False,
        TOP_K=top_k,
        SCALE=weights is not None,
    )

    # Reduce along the top-k dimension, if needed.
    return out.sum(dim=1) if top_k > 1 else out.view(tokens, hidden_size)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_X": 64}, num_warps=2),
        triton.Config({"BLOCK_X": 128}, num_warps=2),
        triton.Config({"BLOCK_X": 256}, num_warps=2),
        triton.Config({"BLOCK_X": 128}, num_warps=4),
        triton.Config({"BLOCK_X": 256}, num_warps=4),
    ],
    key=["NUM_COLUMNS"],
)
@triton.jit
def _binned_copy_wgrad(
    x,
    grad,
    wgrad,
    expert_capacity,
    indices,
    bins,
    NUM_COLUMNS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_X: tl.constexpr,
):
    # Load our indices into the output.
    expert_idx = tl.program_id(0)
    entry_idx = tl.program_id(1)

    # Calculate our offset into the output.
    index_x = expert_idx * expert_capacity + entry_idx

    # Load the index bounds for our bin and calculate
    # the number of tokens assigned to our expert.
    start = 0
    if expert_idx > 0:
        start = tl.load(bins + expert_idx - 1)
    end = tl.load(bins + expert_idx)
    num_tokens = end - start

    # Calculate our offset into the input. If we don't
    # have an input exit early.
    if entry_idx >= num_tokens:
        return
    index_out = tl.load(indices + start + entry_idx)

    # Offset the input and output pointers.
    wgrad += index_out
    grad += tl.multiple_of((index_out // TOP_K) * NUM_COLUMNS, NUM_COLUMNS)
    x += tl.multiple_of(index_x * NUM_COLUMNS, NUM_COLUMNS)
    offsets = tl.max_contiguous(tl.arange(0, BLOCK_X), BLOCK_X)

    acc = tl.zeros((BLOCK_X,), dtype=tl.float32)
    iterations = tl.cdiv(NUM_COLUMNS, BLOCK_X)
    for _ in range(iterations):
        mask = offsets < NUM_COLUMNS
        data = tl.load(x + offsets, mask=mask).to(tl.float32)
        scale = tl.load(grad + offsets, mask=mask).to(tl.float32)
        acc += data * scale
        offsets += BLOCK_X

    # Reduce to get the final result and store.
    out = tl.sum(acc).to(wgrad.dtype.element_ty)
    tl.store(wgrad, out)


def binned_scatter_wgrad(
    x: torch.Tensor, grad: torch.Tensor, indices: torch.Tensor, bins: torch.Tensor, top_k: int
) -> torch.Tensor:
    # Validate the input shapes.
    assert_is_tensor(x, 3)
    assert_is_matrix(grad)
    assert_is_vector(indices)
    assert_is_vector(bins)
    assert_equal(bins.shape[0], x.shape[0])

    num_experts, expert_capacity, hidden_size = x.shape
    tokens = indices.shape[0] // top_k
    out = torch.zeros((tokens * top_k), dtype=x.dtype, device=x.device)
    _binned_copy_wgrad[(num_experts, expert_capacity)](  # type: ignore
        x,
        grad,
        out,
        expert_capacity,
        indices,
        bins,
        NUM_COLUMNS=hidden_size,
        TOP_K=top_k,
    )
    return out
