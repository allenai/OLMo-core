import pytest
import torch
import torch.nn.functional as F

from olmo_core.kernels.swiglu import swiglu_backward_valid_prefix, swiglu_valid_prefix
from olmo_core.testing import requires_gpu, requires_triton


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


def _swiglu_backward_reference(x: torch.Tensor, grad_h: torch.Tensor) -> torch.Tensor:
    hidden = x.shape[-1] // 2
    up = x[:, :hidden]
    gate = x[:, hidden:]
    gate_f32 = gate.to(torch.float32)
    grad_h_f32 = grad_h.to(torch.float32)
    up_f32 = up.to(torch.float32)
    sig = torch.sigmoid(gate_f32)
    silu_gate = gate_f32 * sig
    dsilu = sig * (1.0 + gate_f32 * (1.0 - sig))
    grad_up = grad_h_f32 * silu_gate
    grad_gate = grad_h_f32 * up_f32 * dsilu
    return torch.cat((grad_up, grad_gate), dim=-1).to(dtype=x.dtype)


def test_swiglu_backward_valid_prefix_torch_fallback():
    rows = 5
    hidden = 4
    valid_rows = 3
    x = torch.randn((rows, hidden * 2), dtype=torch.float32)
    grad_h = torch.randn((rows, hidden), dtype=torch.float32)
    num_elements = torch.tensor(valid_rows, dtype=torch.long)
    out = torch.full_like(x, 7.0)

    actual = swiglu_backward_valid_prefix(x, grad_h, num_elements, out=out)
    expected = _swiglu_backward_reference(x[:valid_rows], grad_h[:valid_rows])

    torch.testing.assert_close(actual[:valid_rows], expected)
    torch.testing.assert_close(actual[valid_rows:], torch.full_like(actual[valid_rows:], 7.0))


@requires_gpu
@requires_triton
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
@requires_triton
def test_swiglu_backward_valid_prefix_accepts_device_start_offset():
    rows = 37
    hidden = 64
    start_row = 7
    valid_rows = 19
    x = torch.randn((rows, hidden * 2), device="cuda", dtype=torch.bfloat16)
    grad_h = torch.randn((rows, hidden), device="cuda", dtype=torch.bfloat16)
    start = torch.tensor(start_row, device="cuda", dtype=torch.long)
    num_elements = torch.tensor(valid_rows, device="cuda", dtype=torch.long)
    out = torch.full_like(x, 7.0)

    actual = swiglu_backward_valid_prefix(x, grad_h, num_elements, start=start, out=out)
    expected = _swiglu_backward_reference(
        x[start_row : start_row + valid_rows],
        grad_h[start_row : start_row + valid_rows],
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


@requires_gpu
@requires_triton
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
    expected = x[start_row : start_row + valid_rows, :hidden] * F.silu(
        x[start_row : start_row + valid_rows, hidden:]
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


@requires_gpu
@requires_triton
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("valid_rows", [0, 3, 6])
def test_scoring_swiglu_preserves_eager_rounding(dtype, valid_rows):
    # Stable finite values expose the missing intermediate BF16/FP16 rounding
    # without relying on a broad tolerance that hides the original discrepancy.
    up = torch.tensor([1.5, -2.75, 0.25, 16.0], device="cuda", dtype=dtype)
    gate = torch.tensor([1.0, -1.0, 0.5, -2.0, 3.0, 0.125], device="cuda", dtype=dtype)
    x = torch.cat((up.expand(8, -1), gate.repeat(2)[:8, None].expand(-1, 4)), dim=-1).contiguous()
    out = torch.full((8, 4), 77.0, device="cuda", dtype=dtype)
    start = torch.tensor(1, device="cuda", dtype=torch.long)
    count = torch.tensor(valid_rows, device="cuda", dtype=torch.long)
    result = swiglu_valid_prefix(x, count, start=start, out=out, match_eager_rounding=True)
    eager = x[1 : 1 + valid_rows, :4] * F.silu(x[1 : 1 + valid_rows, 4:])
    torch.testing.assert_close(
        result[1 : 1 + valid_rows], eager, rtol=1e-6 if dtype == torch.float32 else 0, atol=0
    )
    assert torch.equal(result[:1], torch.full_like(result[:1], 77.0))
    assert torch.equal(result[1 + valid_rows :], torch.full_like(result[1 + valid_rows :], 77.0))
    if valid_rows and dtype == torch.bfloat16:
        fused = swiglu_valid_prefix(x, count, start=start)
        assert not torch.equal(fused[1 : 1 + valid_rows], eager)
