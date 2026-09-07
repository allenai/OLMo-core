"""Opt-in real-allocation regressions for large EP SwiGLU address arithmetic."""

import os

import pytest
import torch


@pytest.mark.gpu
@pytest.mark.parametrize("rows", [699_051, 3_145_728])
def test_pairwise_backward_large_offsets(rows):
    """Check doubled-offset overflow and pair-index overflow on an actual large tensor.

    The larger shape exactly matches the failed MB3/capacity8 EP fixture. Every
    output element is checked in bounded chunks, without another full-size
    reference allocation. Explicit opt-in avoids surprising routine GPU CI.
    """
    if os.environ.get("OLMOE3_TEST_LARGE_SWIGLU", "0") != "1":
        pytest.skip("set OLMOE3_TEST_LARGE_SWIGLU=1 for the large-buffer regression")
    assert torch.cuda.is_available(), "Explicit large-buffer gate requires CUDA"
    assert torch.cuda.mem_get_info()[0] >= 60 * 1024**3, "Need at least60 GiB free"
    from olmo_core.ops.swiglu_pairwise import swiglu_backward_pair

    hidden = 1536
    pairs = rows * hidden
    assert pairs * 2 >= 2**31
    x = torch.zeros((rows, hidden * 2), device="cuda", dtype=torch.bfloat16)
    x[:, :hidden].fill_(1)
    dy = torch.ones((rows, hidden), device="cuda", dtype=torch.bfloat16)
    dx = swiglu_backward_pair(x, dy)
    torch.cuda.synchronize()
    # gate=0, up=1, dy=1 gives exactly grad_up=0 and grad_gate=1/2.
    for start in range(0, rows, 16_384):
        chunk = dx[start : start + 16_384]
        assert bool((chunk[:, :hidden] == 0).all())
        assert bool((chunk[:, hidden:] == 0.5).all())
    print(f"LARGE_SWIGLU_PASS rows={rows} pairs={pairs} input_elements={x.numel()}", flush=True)
    del x, dy, dx, chunk
    torch.cuda.empty_cache()
