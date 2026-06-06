import pytest
import torch

from olmo_core.kernels import OlmoMXFP8Tensor


def test_olmo_mxfp8_tensor_roundtrip_and_fallback_dispatch():
    torch.manual_seed(123)
    x = torch.randn(8, 64, dtype=torch.float32)

    mx = OlmoMXFP8Tensor.from_hp(x)

    assert isinstance(mx, OlmoMXFP8Tensor)
    assert mx.shape == x.shape
    assert mx.dtype == x.dtype
    assert mx.qdata.dtype == torch.float8_e4m3fn
    assert mx.scales.dtype == torch.float8_e8m0fnu
    assert mx.scales.shape == (x.shape[0], x.shape[1] // 32)

    x_hat = mx.dequantize(out_dtype=torch.float32)
    assert x_hat.shape == x.shape
    assert torch.isfinite(torch.nan_to_num(x_hat)).all()

    # Generic torch ops fall back to dequantized high-precision tensors.
    y = mx + 1.0
    assert not isinstance(y, OlmoMXFP8Tensor)
    torch.testing.assert_close(y, x_hat + 1.0)


def test_olmo_mxfp8_tensor_can_be_backward_gradient_object():
    seen: dict = {}

    class Upstream(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):  # type: ignore[override]
            del ctx
            return x * 2.0

        @staticmethod
        def backward(ctx, grad_out):  # type: ignore[override]
            del ctx
            seen["grad_type"] = type(grad_out).__name__
            seen["is_mxfp8"] = isinstance(grad_out, OlmoMXFP8Tensor)
            if isinstance(grad_out, OlmoMXFP8Tensor):
                return grad_out.dequantize()
            return torch.zeros_like(grad_out)

    class Downstream(torch.autograd.Function):
        @staticmethod
        def forward(ctx, y):  # type: ignore[override]
            ctx.shape = tuple(y.shape)
            ctx.dtype = y.dtype
            ctx.device = y.device
            return y.sum()

        @staticmethod
        def backward(ctx, grad_out):  # type: ignore[override]
            hp_grad = torch.ones(ctx.shape, dtype=ctx.dtype, device=ctx.device) * grad_out.to(
                ctx.dtype
            )
            return OlmoMXFP8Tensor.from_hp(hp_grad, prefer_triton=False)

    x = torch.randn(4, 64, dtype=torch.bfloat16, requires_grad=True)
    loss = Downstream.apply(Upstream.apply(x))
    loss.backward()

    assert seen == {"grad_type": "OlmoMXFP8Tensor", "is_mxfp8": True}
    assert x.grad is not None
    torch.testing.assert_close(x.grad, torch.ones_like(x))


class _QuantizeMX(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):  # type: ignore[override]
        del ctx
        return OlmoMXFP8Tensor.from_hp(x, prefer_triton=False)

    @staticmethod
    def backward(ctx, grad_out):  # type: ignore[override]
        del ctx
        if isinstance(grad_out, OlmoMXFP8Tensor):
            return grad_out.dequantize()
        return torch.zeros_like(grad_out)


class _ConsumeMX(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mx):  # type: ignore[override]
        if not isinstance(mx, OlmoMXFP8Tensor):
            raise RuntimeError(f"expected OlmoMXFP8Tensor, got {type(mx)!r}")
        ctx.shape = tuple(mx.shape)
        ctx.dtype = mx.dtype
        ctx.device = mx.device
        return mx.dequantize().sum()

    @staticmethod
    def backward(ctx, grad_out):  # type: ignore[override]
        hp_grad = torch.ones(ctx.shape, dtype=ctx.dtype, device=ctx.device) * grad_out.to(ctx.dtype)
        return OlmoMXFP8Tensor.from_hp(hp_grad, prefer_triton=False)


def _mx_loss(x: torch.Tensor) -> torch.Tensor:
    return _ConsumeMX.apply(_QuantizeMX.apply(x))


def test_olmo_mxfp8_tensor_forward_backward_edge():
    x = torch.randn(4, 64, dtype=torch.bfloat16, requires_grad=True)
    loss = _mx_loss(x)
    loss.backward()

    assert x.grad is not None
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_olmo_mxfp8_tensor_torch_compile_eager_backend():
    x = torch.randn(4, 64, dtype=torch.bfloat16, requires_grad=True)
    compiled = torch.compile(_mx_loss, fullgraph=True, backend="eager")

    loss = compiled(x)
    loss.backward()

    assert x.grad is not None
    torch.testing.assert_close(x.grad, torch.ones_like(x))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA inductor smoke test requires CUDA")
def test_olmo_mxfp8_tensor_torch_compile_inductor_cuda_smoke():
    x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    # The custom Functions below use prefer_triton=False so Dynamo/Inductor can
    # trace the prototype through torch ops. The production Triton quant/DQ
    # kernels still need opaque custom-op/fake impl wrapping for fullgraph
    # inductor capture.
    compiled = torch.compile(_mx_loss, fullgraph=True, backend="inductor")

    loss = compiled(x)
    loss.backward()

    assert x.grad is not None
    torch.testing.assert_close(x.grad, torch.ones_like(x))
