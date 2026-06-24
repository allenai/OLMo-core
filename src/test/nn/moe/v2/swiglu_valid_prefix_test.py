import torch
import torch.nn.functional as F

from olmo_core.kernels.swiglu import swiglu_valid_prefix
from olmo_core.testing import requires_gpu


def test_swiglu_valid_prefix_torch_fallback():
    rows = 5
    hidden = 4
    valid_rows = 3
    x = torch.randn((rows, hidden * 2), dtype=torch.float32)
    num_elements = torch.tensor(valid_rows, dtype=torch.long)
    out = torch.full((rows, hidden), 7.0, dtype=torch.float32)

    actual = swiglu_valid_prefix(x, num_elements, out=out)
    expected = x[:valid_rows, :hidden] * F.silu(x[:valid_rows, hidden:])

    torch.testing.assert_close(actual[:valid_rows], expected)
    torch.testing.assert_close(actual[valid_rows:], torch.full_like(actual[valid_rows:], 7.0))


@requires_gpu
def test_swiglu_valid_prefix_matches_torch_and_leaves_tail_untouched():
    rows = 37
    hidden = 64
    valid_rows = 19
    x = torch.randn((rows, hidden * 2), device="cuda", dtype=torch.bfloat16)
    num_elements = torch.tensor(valid_rows, device="cuda", dtype=torch.long)
    out = torch.full((rows, hidden), 7.0, device="cuda", dtype=torch.bfloat16)

    actual = swiglu_valid_prefix(x, num_elements, out=out)
    expected = x[:valid_rows, :hidden] * F.silu(x[:valid_rows, hidden:])

    torch.testing.assert_close(actual[:valid_rows], expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(
        actual[valid_rows:],
        torch.full_like(actual[valid_rows:], 7.0),
        atol=0.0,
        rtol=0.0,
    )


@requires_gpu
def test_swiglu_valid_prefix_accepts_device_start_offset():
    rows = 37
    hidden = 64
    start_row = 7
    valid_rows = 19
    x = torch.randn((rows, hidden * 2), device="cuda", dtype=torch.bfloat16)
    start = torch.tensor(start_row, device="cuda", dtype=torch.long)
    num_elements = torch.tensor(valid_rows, device="cuda", dtype=torch.long)
    out = torch.full((rows, hidden), 7.0, device="cuda", dtype=torch.bfloat16)

    actual = swiglu_valid_prefix(x, num_elements, start=start, out=out)
    expected = (
        x[start_row : start_row + valid_rows, :hidden]
        * F.silu(x[start_row : start_row + valid_rows, hidden:])
    )

    torch.testing.assert_close(
        actual[start_row : start_row + valid_rows],
        expected,
        atol=2e-2,
        rtol=2e-2,
    )
    torch.testing.assert_close(
        actual[:start_row],
        torch.full_like(actual[:start_row], 7.0),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        actual[start_row + valid_rows :],
        torch.full_like(actual[start_row + valid_rows :], 7.0),
        atol=0.0,
        rtol=0.0,
    )
