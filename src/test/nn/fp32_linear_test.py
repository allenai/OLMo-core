import pytest
import torch
from torch.nn import functional as F

from olmo_core.config import DType
from olmo_core.exceptions import OLMoConfigurationError
from olmo_core.nn.fp32_linear import FP32OutputLinear
from olmo_core.nn.lm_head import LMHeadConfig, LMHeadType, LMLossImplementation
from olmo_core.testing import requires_gpu


def test_config_roundtrip_and_unsupported_modes():
    config = LMHeadConfig(fp32_output=True, dtype=DType.bfloat16)
    head = config.build(d_model=16, vocab_size=32)
    assert isinstance(head.w_out, FP32OutputLinear)
    assert head.w_out.weight.dtype == torch.bfloat16
    assert config.as_dict()["fp32_output"] is True
    with pytest.raises(OLMoConfigurationError, match="default LM head"):
        LMHeadConfig(fp32_output=True, name=LMHeadType.normalized).build(d_model=16, vocab_size=32)
    with pytest.raises(OLMoConfigurationError, match="default LM loss"):
        LMHeadConfig(fp32_output=True, loss_implementation=LMLossImplementation.fused_linear).build(
            d_model=16, vocab_size=32
        )
    with pytest.raises(OLMoConfigurationError, match="tensor parallelism"):
        head.apply_tp(None)


def test_cpu_reference_and_state_dict_compatibility():
    torch.manual_seed(17)
    baseline = LMHeadConfig(bias=False).build(d_model=16, vocab_size=32)
    head = LMHeadConfig(fp32_output=True, bias=False).build(d_model=16, vocab_size=32)
    head.load_state_dict(baseline.state_dict(), strict=True)
    x = torch.randn(2, 3, 16, requires_grad=True)
    y = head(x)
    torch.testing.assert_close(y, baseline(x), rtol=0, atol=0)
    y.square().mean().backward()
    assert x.grad is not None and head.w_out.weight.grad is not None
    assert set(head.state_dict()) == set(baseline.state_dict())


@requires_gpu
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("autocast", [False, True])
def test_cuda_forward_and_gradients_against_fp32_reference(bias, autocast):
    torch.manual_seed(29)
    torch.backends.cuda.matmul.allow_tf32 = False
    dtype = torch.float32 if autocast else torch.bfloat16
    layer = FP32OutputLinear(64, 128, bias=bias, device="cuda", dtype=dtype)
    # Noncontiguous input and a frozen/nonfrozen operand are covered below.
    x = (
        torch.randn(2, 64, 7, device="cuda", dtype=dtype)
        .transpose(1, 2)
        .detach()
        .requires_grad_(True)
    )
    xr = x.detach().bfloat16().float().requires_grad_(True)
    wr = layer.weight.detach().bfloat16().float().requires_grad_(True)
    br = None if not bias else layer.bias.detach().bfloat16().float().requires_grad_(True)
    expected = F.linear(xr, wr, br)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        actual = layer(x)
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
    grad = torch.randn_like(actual) / 8
    actual.backward(grad)
    expected.backward(grad)
    # Backward deliberately retains ordinary BF16 GEMMs, so compare against the
    # high-precision derivative with a tolerance for BF16 upstream rounding.
    torch.testing.assert_close(x.grad.float(), xr.grad, atol=0.002, rtol=0.025)
    torch.testing.assert_close(layer.weight.grad.float(), wr.grad, atol=0.012, rtol=0.025)
    if bias:
        torch.testing.assert_close(layer.bias.grad.float(), br.grad, atol=0.003, rtol=0.01)
    original = layer.weight.detach().clone()
    torch.optim.SGD(layer.parameters(), lr=0.1).step()
    assert not torch.equal(original, layer.weight)


@requires_gpu
def test_cuda_compiled_loss_backward_and_frozen_weight():
    torch.manual_seed(31)
    head = LMHeadConfig(fp32_output=True, bias=False, dtype=DType.bfloat16).build(
        d_model=64, vocab_size=128, init_device="cuda"
    )
    eager = torch.randn(2, 5, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    compiled_input = eager.detach().clone().requires_grad_(True)
    labels = torch.randint(0, 128, (2, 5), device="cuda")
    expected = head(eager, labels=labels, return_logits=True)
    expected.loss.backward()
    expected_grad = eager.grad.clone()
    head.zero_grad(set_to_none=True)
    compiled = torch.compile(head, fullgraph=True)
    actual = compiled(compiled_input, labels=labels, return_logits=True)
    actual.loss.backward()
    assert actual.logits.dtype == torch.float32
    torch.testing.assert_close(actual.logits, expected.logits, rtol=1e-5, atol=2e-6)
    torch.testing.assert_close(compiled_input.grad, expected_grad, rtol=0.02, atol=0.0001)
    head.w_out.weight.requires_grad_(False)
    x = eager.detach().clone().requires_grad_(True)
    head(x).sum().backward()
    assert torch.isfinite(x.grad).all()
