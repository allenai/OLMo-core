import torch
import torch.nn.functional as F

import olmo_core.kernels.scaled_grouped_mm as scaled_grouped_mm_module
from olmo_core.kernels.mxfp8_utils import (
    dequantize_rows_from_mxfp8,
    quantize_grouped_2d_to_mxfp8_blocked,
    quantize_grouped_weight_3d_to_mxfp8_blocked,
    quantize_rows_to_mxfp8,
)
from olmo_core.kernels.scaled_grouped_mm import (
    ScaledGroupedMMPrequantizedRHS,
    prequantize_scaled_grouped_mm_rhs,
    scaled_grouped_mm_q,
)


def _build_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(123)
    group_sizes = torch.tensor([3, 2, 4], dtype=torch.int32)
    offs = torch.cumsum(group_sizes, dim=0, dtype=torch.int32)
    a = torch.randn(int(offs[-1].item()), 512, dtype=torch.float32)
    b = torch.randn(group_sizes.numel(), 512, 512, dtype=torch.float32)
    return a, b, offs


def _stub_forward(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    offs: torch.Tensor,
    *,
    use_fast_accum: bool,
    prequantized_lhs=None,
    prequantized_rhs=None,
) -> torch.Tensor:
    del use_fast_accum, prequantized_lhs, prequantized_rhs
    return F.grouped_mm(mat_a, mat_b, offs=offs)


def test_mxfp8_row_quant_dequant_roundtrip():
    x = torch.randn(16, 512, dtype=torch.float32)
    qdata, scales = quantize_rows_to_mxfp8(x, block_size=32)
    assert qdata.dtype == torch.float8_e4m3fn
    assert scales.dtype == torch.float8_e8m0fnu

    x_hat = dequantize_rows_from_mxfp8(qdata, scales, block_size=32, out_dtype=torch.float32)
    assert x_hat.shape == x.shape
    finite = torch.isfinite(x_hat)
    # e4m3fn quantization can produce a small number of non-finite values on CPU.
    assert float(finite.float().mean().item()) > 0.98
    # Float8 round-trip error can be large; keep this as a coarse sanity bound.
    x_hat_safe = torch.nan_to_num(x_hat, nan=0.0, posinf=0.0, neginf=0.0)
    rel_err = (x - x_hat_safe).norm() / x.norm().clamp_min(1e-6)
    assert float(rel_err.item()) < 0.6


def test_mxfp8_row_quant_no_nan_on_cuda():
    if not torch.cuda.is_available():
        return
    x = torch.randn(256, 512, device="cuda", dtype=torch.bfloat16)
    qdata, scales = quantize_rows_to_mxfp8(x, block_size=32)
    assert bool(torch.isfinite(qdata.float()).all())
    assert bool(torch.isfinite(scales.float()).all())


def test_quantize_rows_to_mxfp8_supports_output_buffers():
    x = torch.randn(16, 512, dtype=torch.float32)
    out_q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    out_scales = torch.empty((x.shape[0], x.shape[1] // 32), dtype=torch.float8_e8m0fnu)

    qdata, scales = quantize_rows_to_mxfp8(
        x,
        block_size=32,
        out=out_q,
        scales_out=out_scales,
    )
    assert qdata.untyped_storage().data_ptr() == out_q.untyped_storage().data_ptr()
    assert scales.untyped_storage().data_ptr() == out_scales.untyped_storage().data_ptr()
    assert qdata.shape == x.shape
    assert scales.shape == (x.shape[0], x.shape[1] // 32)


def test_scaled_grouped_mm_q_forward_matches_grouped_mm_reference(monkeypatch):
    monkeypatch.setattr(scaled_grouped_mm_module, "_forward_scaled_grouped_mm_mxfp8", _stub_forward)
    a, b, offs = _build_inputs()

    out = scaled_grouped_mm_q(a, b, offs=offs)
    ref = F.grouped_mm(a, b, offs=offs)
    torch.testing.assert_close(out, ref)


def test_scaled_grouped_mm_q_backward_matches_grouped_mm_reference(monkeypatch):
    monkeypatch.setattr(scaled_grouped_mm_module, "_forward_scaled_grouped_mm_mxfp8", _stub_forward)

    a, b, offs = _build_inputs()
    a = a.detach().requires_grad_(True)
    b = b.detach().requires_grad_(True)

    out = scaled_grouped_mm_q(a, b, offs=offs)
    loss = (out * out).mean()
    grad_a, grad_b = torch.autograd.grad(loss, (a, b))

    a_ref = a.detach().clone().requires_grad_(True)
    b_ref = b.detach().clone().requires_grad_(True)
    out_ref = F.grouped_mm(a_ref, b_ref, offs=offs)
    loss_ref = (out_ref * out_ref).mean()
    grad_a_ref, grad_b_ref = torch.autograd.grad(loss_ref, (a_ref, b_ref))

    torch.testing.assert_close(grad_a, grad_a_ref)
    torch.testing.assert_close(grad_b, grad_b_ref)


def test_scaled_grouped_mm_q_input_grad_out_alias(monkeypatch):
    monkeypatch.setattr(scaled_grouped_mm_module, "_forward_scaled_grouped_mm_mxfp8", _stub_forward)

    a, b, offs = _build_inputs()
    a = a.detach().requires_grad_(True)
    b = b.detach().requires_grad_(True)

    input_grad_out = torch.empty_like(a)

    out = scaled_grouped_mm_q(
        a,
        b,
        offs=offs,
        input_grad_out=input_grad_out,
    )

    grad_a, _grad_b = torch.autograd.grad(
        out,
        (a, b),
        grad_outputs=torch.ones_like(out).contiguous(),
    )
    assert grad_a.untyped_storage().data_ptr() == input_grad_out.untyped_storage().data_ptr()


def test_scaled_grouped_mm_q_empty_input_backward_returns_zero_grads():
    mat_a = torch.empty((0, 512), dtype=torch.float32, requires_grad=True)
    mat_b = torch.randn(3, 512, 512, dtype=torch.float32, requires_grad=True)
    offs = torch.zeros((3,), dtype=torch.int32)

    out = scaled_grouped_mm_q(mat_a, mat_b, offs=offs)
    assert out.shape == (0, 512)
    assert out.dtype == torch.bfloat16

    out.sum().backward()

    assert mat_a.grad is not None
    assert mat_a.grad.shape == mat_a.shape
    assert mat_b.grad is not None
    torch.testing.assert_close(mat_b.grad, torch.zeros_like(mat_b))


def test_scaled_grouped_mm_q_empty_input_backward_uses_input_grad_out():
    mat_a = torch.empty((0, 512), dtype=torch.float32, requires_grad=True)
    mat_b = torch.randn(3, 512, 512, dtype=torch.float32, requires_grad=True)
    offs = torch.zeros((3,), dtype=torch.int32)
    input_grad_out = torch.empty_like(mat_a)

    out = scaled_grouped_mm_q(
        mat_a,
        mat_b,
        offs=offs,
        input_grad_out=input_grad_out,
    )
    (grad_a, _grad_b) = torch.autograd.grad(out.sum(), (mat_a, mat_b))

    assert grad_a.untyped_storage().data_ptr() == input_grad_out.untyped_storage().data_ptr()


def test_quantize_grouped_2d_to_mxfp8_blocked_keeps_capacity_shape():
    x = torch.randn(16, 512, dtype=torch.float32)
    # Active rows are 9, but input has capacity 16.
    offs = torch.tensor([4, 7, 9], dtype=torch.int32)

    qdata, scales_blocked = quantize_grouped_2d_to_mxfp8_blocked(x, offs)
    assert qdata.shape == x.shape
    assert qdata.dtype == torch.float8_e4m3fn

    assert scales_blocked.dtype == torch.float8_e8m0fnu
    assert scales_blocked.shape[1] == x.shape[1] // 32
    assert scales_blocked.shape[0] >= x.shape[0]


def test_quantize_grouped_weight_3d_to_mxfp8_blocked_returns_col_major_layout():
    b = torch.randn(3, 512, 512, dtype=torch.float32)
    qdata, scales_blocked = quantize_grouped_weight_3d_to_mxfp8_blocked(b)

    assert qdata.shape == b.shape
    assert qdata.dtype == torch.float8_e4m3fn
    # scaled_grouped_mm expects transposed RHS memory layout on trailing dims.
    assert qdata.stride(-2) == 1
    assert qdata.stride(-1) == b.shape[-2]
    assert scales_blocked.shape[0] == b.shape[0]
    assert scales_blocked.dtype == torch.float8_e8m0fnu


def test_scaled_grouped_mm_q_forces_mat2_col_major_layout(monkeypatch):
    if not torch.cuda.is_available():
        return
    a, b, offs = _build_inputs()
    a = a.to(device="cuda")
    b = b.to(device="cuda")
    offs = offs.to(device="cuda")

    def _fake_quant_a(mat_a: torch.Tensor, *, block_size: int = 32):
        del block_size
        return mat_a.to(torch.float8_e4m3fn), torch.ones(
            (mat_a.shape[0], mat_a.shape[1] // 32),
            dtype=torch.float8_e8m0fnu,
            device=mat_a.device,
        )

    def _fake_quant_b(mat_b: torch.Tensor, *, block_size: int = 32):
        del block_size
        # Return row-major data intentionally; wrapper should convert it.
        q = mat_b.to(torch.float8_e4m3fn).contiguous()
        s = torch.ones(
            (mat_b.shape[0], mat_b.shape[2], mat_b.shape[1] // 32),
            dtype=torch.float8_e8m0fnu,
            device=mat_b.device,
        )
        return q, s

    seen = {}

    def _fake_scaled_grouped_mm_v2_cuda(
        mat_a_q: torch.Tensor,
        mat_b_q: torch.Tensor,
        scale_a_unblocked: torch.Tensor,
        scale_b_unblocked: torch.Tensor,
        *,
        offs: torch.Tensor,
        use_fast_accum: bool,
    ):
        del mat_a_q, scale_a_unblocked, scale_b_unblocked, offs, use_fast_accum
        seen["shape"] = tuple(mat_b_q.shape)
        seen["stride"] = tuple(mat_b_q.stride())
        return torch.zeros(
            (a.shape[0], b.shape[-1]),
            device=mat_b_q.device,
            dtype=torch.bfloat16,
        )

    monkeypatch.setattr(
        scaled_grouped_mm_module,
        "quantize_rows_to_mxfp8",
        _fake_quant_a,
    )
    monkeypatch.setattr(
        scaled_grouped_mm_module,
        "quantize_grouped_weight_3d_to_mxfp8_blocked",
        _fake_quant_b,
    )
    monkeypatch.setattr(
        scaled_grouped_mm_module,
        "_scaled_grouped_mm_v2_cuda",
        _fake_scaled_grouped_mm_v2_cuda,
    )

    _ = scaled_grouped_mm_module._forward_scaled_grouped_mm_mxfp8(
        a,
        b,
        offs,
        use_fast_accum=True,
    )

    assert seen["shape"] == tuple(b.shape)
    # trailing 2D strides must be column-major: (.., 1, K)
    assert seen["stride"][-2] == 1
    assert seen["stride"][-1] == b.shape[-2]


def test_scaled_grouped_mm_q_uses_prequantized_rhs(monkeypatch):
    if not torch.cuda.is_available():
        return
    torch.manual_seed(123)
    offs = torch.tensor([3, 5, 9], dtype=torch.int32, device="cuda")
    a = torch.randn(int(offs[-1].item()), 512, dtype=torch.float32, device="cuda")
    b = torch.randn(offs.numel(), 512, 512, dtype=torch.float32, device="cuda")
    preq = prequantize_scaled_grouped_mm_rhs(b)
    assert isinstance(preq, ScaledGroupedMMPrequantizedRHS)

    called = {"rhs_quant": 0}

    def _forbidden_rhs_quant(_mat_b, *, block_size: int = 32):
        del block_size
        called["rhs_quant"] += 1
        raise AssertionError(
            "RHS quantization should be bypassed when prequantized_rhs is provided"
        )

    monkeypatch.setattr(
        scaled_grouped_mm_module,
        "quantize_grouped_weight_3d_to_mxfp8_blocked",
        _forbidden_rhs_quant,
    )

    def _fake_scaled_grouped_mm_v2_cuda(
        mat_a_q: torch.Tensor,
        mat_b_q: torch.Tensor,
        scale_a_unblocked: torch.Tensor,
        scale_b_unblocked: torch.Tensor,
        *,
        offs: torch.Tensor,
        use_fast_accum: bool,
    ):
        del mat_a_q, scale_a_unblocked, scale_b_unblocked, offs, use_fast_accum
        # Ensure the prequantized RHS object is the one actually consumed.
        assert mat_b_q.untyped_storage().data_ptr() == preq.mat_b_q.untyped_storage().data_ptr()
        return torch.zeros((a.shape[0], b.shape[-1]), dtype=torch.bfloat16, device=a.device)

    monkeypatch.setattr(
        scaled_grouped_mm_module,
        "_scaled_grouped_mm_v2_cuda",
        _fake_scaled_grouped_mm_v2_cuda,
    )
    out = scaled_grouped_mm_q(
        a,
        b,
        offs=offs,
        prequantized_rhs=preq,
    )
    assert out.shape == (a.shape[0], b.shape[-1])
    assert called["rhs_quant"] == 0


def test_scaled_grouped_mm_q_backward_uses_prequantized_lhs_when_mat_a_not_saved(monkeypatch):
    torch.manual_seed(123)
    offs = torch.tensor([3, 5, 9], dtype=torch.int32)
    mat_a_real = torch.randn(int(offs[-1].item()), 512, dtype=torch.float32)
    mat_a_bad = torch.zeros_like(mat_a_real)
    mat_b = torch.randn(offs.numel(), 512, 512, dtype=torch.float32, requires_grad=True)
    mat_a_q, mat_a_s = quantize_rows_to_mxfp8(mat_a_real, block_size=32)
    preq_lhs = scaled_grouped_mm_module.ScaledGroupedMMPrequantizedLHS(
        mat_a_q=mat_a_q,
        scale_a=mat_a_s,
        mat_a_shape=tuple(mat_a_bad.shape),
        scales_are_blocked=False,
    )

    def _stub_forward(
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        offs: torch.Tensor,
        *,
        use_fast_accum: bool,
        prequantized_lhs=None,
        prequantized_rhs=None,
    ) -> torch.Tensor:
        del use_fast_accum, prequantized_lhs, prequantized_rhs
        return F.grouped_mm(mat_a, mat_b, offs=offs)

    monkeypatch.setattr(scaled_grouped_mm_module, "_forward_scaled_grouped_mm_mxfp8", _stub_forward)

    captured = {}

    def _fake_grouped_mm(mat_a: torch.Tensor, mat_b: torch.Tensor, *, offs: torch.Tensor):
        # Backward grad_b path: mat_b is reconstructed lhs (rank-2 [M, K]).
        if mat_b.ndim == 2 and tuple(mat_b.shape) == tuple(mat_a_bad.shape):
            captured["lhs_for_grad_b"] = mat_b.clone()
            return torch.zeros((offs.numel(), mat_a.shape[0], mat_b.shape[-1]), dtype=mat_a.dtype)
        return torch.zeros((mat_a.shape[0], mat_b.shape[-1]), dtype=mat_a.dtype)

    monkeypatch.setattr(F, "grouped_mm", _fake_grouped_mm)

    out = scaled_grouped_mm_q(
        mat_a_bad,
        mat_b,
        offs=offs,
        prequantized_lhs=preq_lhs,
    )
    loss = out.sum()
    (grad_b,) = torch.autograd.grad(loss, (mat_b,))
    assert grad_b is not None

    assert "lhs_for_grad_b" in captured
    mat_a_expected = dequantize_rows_from_mxfp8(
        mat_a_q, mat_a_s, block_size=32, out_dtype=out.dtype
    )
    torch.testing.assert_close(captured["lhs_for_grad_b"], mat_a_expected)
