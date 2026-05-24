import pytest
import torch
import torch.nn.functional as F

from olmo_core.nn.mxfp8_linear import MXFP8Linear


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("save_wgrad_input", ["mxfp8", "bf16"])
def test_mxfp8_linear_forward_backward_matches_reference_loosely(save_wgrad_input):
    torch.manual_seed(1256)
    layer = MXFP8Linear(
        64,
        96,
        bias=True,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
        save_wgrad_input=save_wgrad_input,
    )
    with torch.no_grad():
        layer.weight.mul_(0.05)
        assert layer.bias is not None
        layer.bias.mul_(0.05)

    x = (torch.randn(64, 64, device="cuda", dtype=torch.bfloat16) * 0.05).requires_grad_()
    x_ref = x.detach().clone().float().requires_grad_()
    weight_ref = layer.weight.detach().clone().float().requires_grad_()
    bias_ref = layer.bias.detach().clone().float().requires_grad_()
    grad = torch.randn(64, 96, device="cuda", dtype=torch.bfloat16) * 0.05

    out = layer(x)
    out_ref = F.linear(x_ref, weight_ref, bias_ref)
    out.backward(grad)
    out_ref.backward(grad.float())

    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), out_ref, atol=2e-3, rtol=0.0)
    torch.testing.assert_close(x.grad.float(), x_ref.grad, atol=3e-3, rtol=0.0)
    assert layer.weight.grad is None
    assert layer.fp8_weight_store.grad_bf16 is not None
    torch.testing.assert_close(
        layer.fp8_weight_store.grad_bf16.float(), weight_ref.grad, atol=2e-2, rtol=0.0
    )
    torch.testing.assert_close(layer.bias.grad.float(), bias_ref.grad, atol=5e-3, rtol=0.0)


def test_mxfp8_linear_accumulates_weight_grad_in_store():
    torch.manual_seed(3852)
    layer = MXFP8Linear(
        64,
        96,
        bias=False,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )
    layer.refresh_mxfp8_cache()
    layer.release_mxfp8_anchor_storage()

    x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = layer(x)
    out.float().sum().backward()

    assert layer.weight.grad is None
    assert x.grad is not None
    assert x.grad.shape == x.shape
    store = layer.fp8_weight_store
    assert store.grad_bf16 is not None
    assert store.grad_bf16.shape == layer.weight.shape
    assert store.grad_bf16.dtype == torch.bfloat16


def test_mxfp8_linear_training_asserts_unaligned_wgrad_reduction():
    layer = MXFP8Linear(
        64,
        96,
        bias=False,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )
    x = torch.randn(33, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    with pytest.raises(AssertionError, match="wgrad reduction dim"):
        layer(x)
