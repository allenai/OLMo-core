from __future__ import annotations

import pytest
import torch

from olmo_core.kernels.grouped_mm_row_offset import grouped_mm_row_offset


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_grouped_mm_row_offset_writes_only_shifted_groups():
    device = torch.device("cuda")
    torch.manual_seed(1234)

    capacity = 64
    k = 32
    n = 48
    counts = torch.tensor([8, 16, 8], device=device, dtype=torch.int32)
    row_start = torch.tensor(24, device=device, dtype=torch.int64)
    a = torch.randn(capacity, k, device=device, dtype=torch.bfloat16)
    b = torch.randn(counts.numel(), k, n, device=device, dtype=torch.bfloat16)
    out = torch.full((capacity, n), -7.0, device=device, dtype=torch.bfloat16)

    grouped_mm_row_offset(a, b, counts, row_start=row_start, out=out)
    torch.cuda.synchronize()

    expected = torch.full_like(out, -7.0)
    start = int(row_start.item())
    cursor = start
    for group_idx, count in enumerate(counts.cpu().tolist()):
        expected[cursor : cursor + count] = a[cursor : cursor + count] @ b[group_idx]
        cursor += count

    torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)
