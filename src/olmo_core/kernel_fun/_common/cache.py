"""The CuTe call cache: marshal once, poke the pointer, launch.

Every CuTe kernel in this package drives its compiled object through the same three
mechanisms, and they live here once rather than copy-pasted per kernel (which is how they
exist in the research tree, and how the keepalive bug below survived in four places at
once):

  cute_view        — a torch tensor as a cute.Tensor with the mode order cute wants and the
                     batch-ish modes marked dynamic, so one compile serves every B/T.
  retarget         — rewrite the descriptor's data pointer for this call. Marshaling costs
                     ~0.07ms per kernel; a ctypes word-write costs nothing.
  release_keepalives — drop the DLPack reference to the tensor that exported the view.

The last one is not an optimization, it is a leak fix: `from_dlpack` requires the consumer
to keep the exporter alive, so a cached view pins the FIRST call's tensor for the life of
the process. Measured on the kda chain at B8/T1024/H8/HV8/K128/V256: 778 MiB pinned across
its four cache entries, ~24 GiB at prod8192 shapes, never freed — which is exactly the
condition that pushes the caching allocator into device-synchronizing cudaMalloc retries.
gnorm hit the same bug at +3.5 GiB/rank on the 1.4b ladder.

IMPORTANT for anything cached here: outputs are allocated fresh per call and retargeted,
never reused. A cache-owned output silently aliases across calls, which a bench that makes
one call per iteration cannot see.
"""

from __future__ import annotations

import ctypes

import torch
from cutlass.cute.runtime import from_dlpack


def cute_view(t: torch.Tensor, perm: tuple[int, ...], dyn_modes: tuple[int, ...]):
    """A cute.Tensor view of `t`, permuted to `perm`, with `dyn_modes` dynamic.

    cute wants modes ordered outermost-to-innermost and checks that against the strides.
    `tt.dim_order()` is not usable: it places size-1 modes by contiguity heuristics, so e.g.
    (512,128,2,1):(256,1,128,131072) comes back as (0,2,3,1) and cute rejects it. Sorting
    the permuted tensor's strides has the mirror problem — size-1 modes tie with their
    neighbour and land on the wrong side. Take the order from the UNPERMUTED tensor, where
    descending stride is unambiguous, and map it through perm.

    detach: inside an autograd.Function the incoming leaves still carry requires_grad and
    dlpack refuses to export those. Every gradient path here is hand-written, so only the
    storage is wanted.
    """
    t = t.detach()
    base_order = sorted(range(t.dim()), key=lambda i: -t.stride(i))
    new_of_old = {old: new for new, old in enumerate(perm)}
    stride_order = tuple(new_of_old[d] for d in base_order)
    ct = from_dlpack(t.permute(*perm), assumed_align=16)
    for m in dyn_modes:
        ct = ct.mark_compact_shape_dynamic(mode=m, stride_order=stride_order)
    return ct


def retarget(ct, t: torch.Tensor) -> None:
    """Point a cached view at this call's tensor. One 64-bit store."""
    ptr = t.data_ptr()
    assert ptr & 15 == 0, "kernel views assume 16B alignment"
    ctypes.c_uint64.from_address(ct.__c_pointers__()[0]).value = ptr


def release_keepalives(*cts) -> None:
    """Drop cached views' DLPack references to the tensors that exported them.

    Call this after `cute.compile` and before the entry lands in a call cache, on EVERY
    view including the outputs' — a cached entry needs only the C memref descriptor, which
    the view owns, since `retarget` rewrites the data word before every launch.

    Populate the pointer cache first, then null the two reference fields. Those names are
    cutlass internals: fail loudly if an upgrade renames them rather than silently going
    back to pinning gigabytes.
    """
    for ct in cts:
        ct.__c_pointers__()  # materialize _c_pointers_cache before the source goes away
        d = ct.__dict__
        assert "_dlpack_data" in d and "_dltensor_wrapper" in d, (
            "cutlass.cute.runtime._Tensor changed its keepalive fields; re-derive "
            "release_keepalives against the new version (see kernel_fun._common.cache)"
        )
        d["_dlpack_data"] = None
        d["_dltensor_wrapper"] = None


def out_specs(*tensors: torch.Tensor) -> tuple:
    """(shape, dtype) per output — what a cache entry keeps instead of the tensors."""
    return tuple((tuple(t.shape), t.dtype) for t in tensors)


def alloc_outs(specs: tuple, device) -> tuple:
    return tuple(torch.empty(shape, device=device, dtype=dtype) for shape, dtype in specs)


def stream_handle() -> int:
    """The current stream's raw handle — part of every call key, so a stream switch makes a
    new entry rather than launching a stale one."""
    return torch.cuda.current_stream().cuda_stream
