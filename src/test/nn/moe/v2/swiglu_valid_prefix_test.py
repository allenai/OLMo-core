import torch
import torch.nn.functional as F

from olmo_core.kernels.swiglu import swiglu_backward_valid_prefix, swiglu_valid_prefix
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
