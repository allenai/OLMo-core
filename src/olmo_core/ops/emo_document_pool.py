"""Profiling-only document pool selection; no training callers until qualified.

Segment IDs must come from segment_ids_from_eos: contiguous, nondecreasing,
zero-based within each sequence. Pool sizes must be constant within a document.
Use the SAME FP32 scatter addition and pool RNG as the reference. Only eliminate
sorting identical broadcast rows; preserve Inductor's unstable-sort tie policy.
"""

import torch
import triton
import triton.language as tl
from torch._inductor.runtime import triton_helpers


@triton.jit
def _document_pool_mask(sums, pools, segments, output, T: tl.constexpr, E: tl.constexpr):
    batch = tl.program_id(0)
    document = tl.program_id(1)
    last_document = tl.load(segments + batch * T + T - 1)
    if document <= last_document:
        offsets = tl.arange(0, E)
        scores = tl.load(sums + (batch * T + document) * E + offsets)
        _, order = triton_helpers.sort_with_index(
            scores, offsets.to(tl.int64), E, descending=True, stable=False
        )
        pool = tl.load(pools + batch * T + document)
        tl.store(output + (batch * T + document) * E + order, offsets < pool)


def document_pool_keep_mask(scores, segment_ids, pool_sizes):
    """Return the reference per-token mask while sorting only live document rows."""
    if (
        not scores.is_cuda
        or scores.dtype != torch.float32
        or scores.ndim != 3
        or scores.shape[-1] != 512
        or not scores.is_contiguous()
        or not segment_ids.is_contiguous()
        or segment_ids.shape != scores.shape[:2]
        or pool_sizes.shape != segment_ids.shape
    ):
        raise ValueError(
            "Expected contiguous CUDA FP32 [B,T,512] scores and [B,T] segment/pool IDs"
        )
    batch, length, experts = scores.shape
    indices = segment_ids.unsqueeze(-1).expand(-1, -1, experts)
    sums = torch.zeros_like(scores)
    sums.scatter_add_(1, indices, scores)
    # Repeated destinations all contain the same integer pool size. Preserve the
    # caller's RNG draws instead of sampling only the compacted live documents.
    document_pools = torch.zeros_like(pool_sizes)
    document_pools.scatter_(1, segment_ids, pool_sizes)
    document_mask = torch.empty_like(scores, dtype=torch.bool)
    _document_pool_mask[(batch, length)](
        sums, document_pools, segment_ids, document_mask, length, experts, num_warps=4
    )
    # Unwritten slots correspond only to nonexistent documents and are never read.
    return document_mask.gather(1, indices)
