"""Direct arithmetic and backend contracts for BF16-rounded weight gradients."""

import pytest
import torch
import torch.nn.functional as F

from olmo_core.ops import rounded_wgrad
from olmo_core.testing import requires_gpu


def test_rounded_wgrad_rejects_unsupported_backend(monkeypatch):
    rounded_wgrad._compile.cache_clear()
    with pytest.raises(RuntimeError, match="Blackwell"):
        rounded_wgrad._compile((9, 0), ())
    monkeypatch.setattr(rounded_wgrad.importlib.metadata, "version", lambda _: "0.6.0")
    with pytest.raises(RuntimeError, match="quack-kernels==0.5.0"):
        rounded_wgrad._compile((10, 3), ())
    monkeypatch.setattr(rounded_wgrad.importlib.metadata, "version", lambda _: "0.5.0")
    monkeypatch.setattr(torch, "__version__", "2.14.0")
    with pytest.raises(RuntimeError, match="Torch 2.11 or 2.13"):
        rounded_wgrad._compile((10, 3), ())


@requires_gpu
@pytest.mark.parametrize("counts", [(0, 0, 0, 0), (1, 15, 0, 65), (64, 64, 64, 64)])
@pytest.mark.parametrize("m,n", [(64, 128), (128, 64)])
def test_rounded_wgrad_accumulation(counts, m, n):
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("requires Blackwell")
    torch.manual_seed(319)
    offsets = torch.tensor((0, *counts), device="cuda", dtype=torch.int32).cumsum(
        0, dtype=torch.int32
    )
    expected = torch.randn(len(counts), m, n, device="cuda", dtype=torch.float32)
    actual = expected.clone()
    # Preserve each microbatch's BF16 rounding before accumulating into FP32;
    # a plain FP32 GEMM+add computes a different optimizer update.
    for _ in range(8):
        a = torch.randn(sum(counts), m, device="cuda", dtype=torch.bfloat16).T
        b = torch.randn(sum(counts), n, device="cuda", dtype=torch.bfloat16).T
        expected.add_(F.grouped_mm(a, b.T, offs=offsets[1:]))
        rounded_wgrad.rounded_wgrad_add(a, b, actual, offsets)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
