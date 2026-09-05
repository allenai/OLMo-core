"""Native CUDA top16 routing indices, including boundaries and special values."""

import pytest
import torch

from olmo_core.ops.moe_top16 import top16_native_indices


@pytest.mark.gpu
def test_top16_native_contract():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(182)
    native = torch.compile(lambda x: x.topk(16, dim=-1).indices, fullgraph=True)
    candidate = torch.compile(top16_native_indices, fullgraph=True)
    for rows in (1, 17, 514, 32768):
        for case in ("random", "tied", "zeros", "signed_zero", "negative", "nonfinite"):
            x = torch.rand(rows, 512, device="cuda")
            if case == "tied":
                x = (x * 8).floor()
            elif case == "zeros":
                x.zero_()
                x[:, :200] = float("-inf")
            elif case == "signed_zero":
                x.zero_()
                x[:, :300] = -0.0
            elif case == "negative":
                x = x * 8 - 4
            elif case == "nonfinite":
                x[:, :10] = float("nan")
                x[:, 10:20] = float("inf")
                x[:, 20:200] = float("-inf")
            expected, actual = native(x), candidate(x)
            torch.testing.assert_close(expected, actual, rtol=0, atol=0, msg=f"{rows}/{case}")
