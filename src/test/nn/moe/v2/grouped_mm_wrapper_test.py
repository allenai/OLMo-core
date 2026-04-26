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
