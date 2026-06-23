import torch
import torch.nn.functional as F

from olmo_core.kernels.grouped_mm import grouped_mm
from olmo_core.testing import requires_gpu


def _build_grouped_mm_inputs(
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(41)
    group_sizes = torch.tensor([3, 2, 3], device=device, dtype=torch.int32)
    offs = torch.cumsum(group_sizes, dim=0, dtype=torch.int32)
    a = torch.randn(int(offs[-1].item()), 512, device=device, dtype=torch.float32)
    b = torch.randn(group_sizes.numel(), 512, 512, device=device, dtype=torch.float32)
    return a, b, offs


@requires_gpu
def test_grouped_mm_wrapper_matches_torch_grouped_mm():
    device = torch.device("cuda")
    a, b, offs = _build_grouped_mm_inputs(device=device)

    out = grouped_mm(a, b, offs=offs)
    ref = F.grouped_mm(a, b, offs=offs)

    torch.testing.assert_close(out, ref)


@requires_gpu
def test_grouped_mm_wrapper_writes_forward_out_buffer():
    device = torch.device("cuda")
    a, b, offs = _build_grouped_mm_inputs(device=device)
    out_buffer = torch.empty((a.shape[0], b.shape[-1]), device=device, dtype=a.dtype)

    out = grouped_mm(a, b, offs=offs, out=out_buffer)
    ref = F.grouped_mm(a, b, offs=offs)

    assert out.untyped_storage().data_ptr() == out_buffer.untyped_storage().data_ptr()
    torch.testing.assert_close(out, ref)


@requires_gpu
def test_grouped_mm_wrapper_writes_input_grad_out_buffer():
    device = torch.device("cuda")
    a, b, offs = _build_grouped_mm_inputs(device=device)
    a = a.detach().requires_grad_(True)
    b = b.detach().requires_grad_(True)
    input_grad_out = torch.empty_like(a)

    out = grouped_mm(a, b, offs=offs, input_grad_out=input_grad_out)
    grad_out = torch.randn_like(out)
    grad_a, grad_b = torch.autograd.grad(out, (a, b), grad_outputs=grad_out)

    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)
    out_ref = F.grouped_mm(a_ref, b_ref, offs=offs)
    grad_a_ref, grad_b_ref = torch.autograd.grad(out_ref, (a_ref, b_ref), grad_outputs=grad_out)

    assert grad_a.untyped_storage().data_ptr() == input_grad_out.untyped_storage().data_ptr()
    torch.testing.assert_close(grad_a, grad_a_ref)
    torch.testing.assert_close(grad_b, grad_b_ref)


@requires_gpu
def test_grouped_mm_wrapper_backward_ignores_rows_after_offsets():
    device = torch.device("cuda")
    torch.manual_seed(43)
    group_sizes = torch.tensor([3, 2, 3], device=device, dtype=torch.int32)
    offs = torch.cumsum(group_sizes, dim=0, dtype=torch.int32)
    valid_rows = int(offs[-1].item())
    capacity_rows = valid_rows + 5

    a_prefix = torch.randn(valid_rows, 128, device=device, dtype=torch.float32)
    b = torch.randn(group_sizes.numel(), 128, 96, device=device, dtype=torch.float32)
    grad_valid = torch.randn(valid_rows, 96, device=device, dtype=torch.float32)

    def run_with_tail(tail_value: float):
        a = torch.empty(capacity_rows, a_prefix.shape[1], device=device, dtype=torch.float32)
        a[:valid_rows].copy_(a_prefix)
        a[valid_rows:].fill_(tail_value)
        a.requires_grad_(True)
        b_local = b.detach().clone().requires_grad_(True)
        input_grad_out = torch.empty_like(a)
        out = grouped_mm(a, b_local, offs=offs, input_grad_out=input_grad_out)
        grad_a, grad_b = torch.autograd.grad(out[:valid_rows], (a, b_local), grad_valid)
        assert grad_a.untyped_storage().data_ptr() == input_grad_out.untyped_storage().data_ptr()
        return out[:valid_rows].detach(), grad_a[:valid_rows].detach(), grad_b.detach()

    out_a, grad_a_a, grad_b_a = run_with_tail(123.0)
    out_b, grad_a_b, grad_b_b = run_with_tail(-321.0)

    compact_a = a_prefix.detach().clone().requires_grad_(True)
    compact_b = b.detach().clone().requires_grad_(True)
    compact_out = grouped_mm(compact_a, compact_b, offs=offs)
    compact_grad_a, compact_grad_b = torch.autograd.grad(
        compact_out,
        (compact_a, compact_b),
        grad_valid,
    )

    torch.testing.assert_close(out_a, compact_out)
    torch.testing.assert_close(out_b, compact_out)
    torch.testing.assert_close(grad_a_a, compact_grad_a)
    torch.testing.assert_close(grad_a_b, compact_grad_a)
    torch.testing.assert_close(grad_b_a, compact_grad_b)
    torch.testing.assert_close(grad_b_b, compact_grad_b)
