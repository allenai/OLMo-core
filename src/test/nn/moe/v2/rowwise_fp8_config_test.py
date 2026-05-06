from types import SimpleNamespace

import pytest
import torch

from olmo_core.config import DType
from olmo_core.kernels.scaled_grouped_mm import ScaledGroupedMMPrequantizedRHS
from olmo_core.nn.moe.v2.fp8 import (
    MoERowwiseFP8Config,
    normalize_rowwise_fp8_config,
    refresh_rowwise_fp8_cache,
)
from olmo_core.nn.moe.v2.routed_experts import (
    RoutedExperts,
    _swiglu_backward_grad_up_gate_impl,
    swiglu_backward_grad_up_gate,
)


def test_rowwise_fp8_config_validate_block_size():
    cfg = MoERowwiseFP8Config(enabled=True, block_size=16)
    with pytest.raises(ValueError, match="block_size=32"):
        cfg.validate()


def test_rowwise_fp8_config_fail_closed_on_sm90(monkeypatch):
    cfg = MoERowwiseFP8Config(enabled=True, block_size=32)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device=0: (9, 0))

    with pytest.raises(RuntimeError, match="pre-SM100"):
        cfg.assert_runtime_supported()


def test_rowwise_fp8_config_requires_scaled_grouped_mm(monkeypatch):
    cfg = MoERowwiseFP8Config(enabled=True, block_size=32)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device=0: (10, 0))
    monkeypatch.delattr(torch.nn.functional, "scaled_grouped_mm", raising=False)

    with pytest.raises(RuntimeError, match="scaled_grouped_mm"):
        cfg.assert_runtime_supported()


def test_rowwise_fp8_normalize_from_dict():
    cfg = normalize_rowwise_fp8_config({"enabled": True, "block_size": 32})
    assert isinstance(cfg, MoERowwiseFP8Config)
    assert cfg.enabled is True
    assert cfg.block_size == 32


def test_routed_experts_accepts_rowwise_fp8_dict():
    module = RoutedExperts(
        d_model=512,
        hidden_size=1024,
        num_experts=4,
        bias=False,
        dtype=DType.float32,
        rowwise_fp8={"enabled": True, "block_size": 32},
        init_device="cpu",
    )
    assert isinstance(module.rowwise_fp8, MoERowwiseFP8Config)
    assert module.rowwise_fp8 is not None and module.rowwise_fp8.enabled is True


def test_block_refresh_rowwise_fp8_cache_honors_routed_config_without_block_config():
    seen = {"refresh": 0, "invalidate": 0}

    class _FakeRoutedExperts:
        rowwise_fp8 = MoERowwiseFP8Config(enabled=True)

        def refresh_rowwise_fp8_cache(self) -> None:
            seen["refresh"] += 1

        def invalidate_rowwise_fp8_cache(self) -> None:
            seen["invalidate"] += 1

    block = SimpleNamespace(
        rowwise_fp8=None,
        routed_experts=_FakeRoutedExperts(),
        shared_experts=None,
    )

    refresh_rowwise_fp8_cache(block)

    assert seen == {"refresh": 1, "invalidate": 0}
    assert block._shared_rowwise_fp8_up_prequant is None
    assert block._shared_rowwise_fp8_down_prequant is None


def test_block_refresh_rowwise_fp8_cache_disabled_invalidates_without_refresh():
    seen = {"refresh": 0, "invalidate": 0}

    class _FakeRoutedExperts:
        rowwise_fp8 = MoERowwiseFP8Config(enabled=False)

        def refresh_rowwise_fp8_cache(self) -> None:
            seen["refresh"] += 1

        def invalidate_rowwise_fp8_cache(self) -> None:
            seen["invalidate"] += 1

    block = SimpleNamespace(
        rowwise_fp8=MoERowwiseFP8Config(enabled=False),
        routed_experts=_FakeRoutedExperts(),
        shared_experts=None,
        _shared_rowwise_fp8_up_prequant=object(),
        _shared_rowwise_fp8_down_prequant=object(),
        _shared_rowwise_fp8_up_prequant_t=object(),
        _shared_rowwise_fp8_down_prequant_t=object(),
        _shared_rowwise_fp8_weight_versions=(1, 2),
    )

    refresh_rowwise_fp8_cache(block)

    assert seen == {"refresh": 0, "invalidate": 1}
    assert block._shared_rowwise_fp8_up_prequant is None
    assert block._shared_rowwise_fp8_down_prequant is None
    assert block._shared_rowwise_fp8_up_prequant_t is None
    assert block._shared_rowwise_fp8_down_prequant_t is None
    assert block._shared_rowwise_fp8_weight_versions is None


def test_block_refresh_rowwise_fp8_cache_shared_only_does_not_refresh_routed():
    seen = {"refresh": 0, "invalidate": 0}

    class _FakeRoutedExperts:
        rowwise_fp8 = MoERowwiseFP8Config(enabled=False)

        def refresh_rowwise_fp8_cache(self) -> None:
            seen["refresh"] += 1

        def invalidate_rowwise_fp8_cache(self) -> None:
            seen["invalidate"] += 1

    block = SimpleNamespace(
        rowwise_fp8=MoERowwiseFP8Config(enabled=True),
        routed_experts=_FakeRoutedExperts(),
        shared_experts=None,
    )

    refresh_rowwise_fp8_cache(block)

    assert seen == {"refresh": 0, "invalidate": 1}
    assert block._shared_rowwise_fp8_up_prequant is None
    assert block._shared_rowwise_fp8_down_prequant is None


def test_routed_experts_refresh_marks_owned_prequant_caches_versionless(monkeypatch):
    if not torch.cuda.is_available():
        return

    module = RoutedExperts(
        d_model=512,
        hidden_size=256,
        num_experts=2,
        bias=False,
        dtype=DType.float32,
        rowwise_fp8={"enabled": True, "block_size": 32},
        init_device="cuda",
    )

    def _fake_prequant(mat_b: torch.Tensor, *, check_mat_b_version: bool = True):
        return ScaledGroupedMMPrequantizedRHS(
            mat_b_q=torch.empty_like(mat_b, dtype=torch.float8_e4m3fn),
            scale_b=torch.empty((1,), dtype=torch.float8_e8m0fnu),
            mat_b_shape=tuple(mat_b.shape),
            mat_b_version=0 if check_mat_b_version else -1,
        )

    monkeypatch.setattr(
        "olmo_core.nn.moe.v2.routed_experts.prequantize_scaled_grouped_mm_rhs",
        _fake_prequant,
    )

    module.refresh_rowwise_fp8_cache()

    assert module._rowwise_fp8_up_gate_prequant is not None
    assert module._rowwise_fp8_up_gate_prequant.mat_b_version == -1
    assert module._rowwise_fp8_up_gate_prequant_t is not None
    assert module._rowwise_fp8_up_gate_prequant_t.mat_b_version == -1
    assert module._rowwise_fp8_down_prequant is not None
    assert module._rowwise_fp8_down_prequant.mat_b_version == -1
    assert module._rowwise_fp8_down_prequant_t is not None
    assert module._rowwise_fp8_down_prequant_t.mat_b_version == -1


def test_routed_experts_forward_rowwise_fp8_uses_cached_prequantized_rhs(monkeypatch):
    module = RoutedExperts(
        d_model=512,
        hidden_size=256,
        num_experts=2,
        bias=False,
        dtype=DType.float32,
        rowwise_fp8={"enabled": True, "block_size": 32},
        init_device="cpu",
    )

    up_q = module.w_up_gate.transpose(1, 2).to(torch.float8_e4m3fn)
    up_t_q = module.w_up_gate.to(torch.float8_e4m3fn)
    up_s = torch.ones((module.num_experts, 1), dtype=torch.float8_e8m0fnu)
    down_q = module.w_down.to(torch.float8_e4m3fn)
    down_t_q = module.w_down.transpose(1, 2).to(torch.float8_e4m3fn)
    down_s = torch.ones((module.num_experts, 1), dtype=torch.float8_e8m0fnu)

    module._rowwise_fp8_up_gate_prequant = ScaledGroupedMMPrequantizedRHS(
        mat_b_q=up_q,
        scale_b=up_s,
        mat_b_shape=tuple(module.w_up_gate.transpose(1, 2).shape),
        mat_b_version=int(module.w_up_gate._version),
    )
    module._rowwise_fp8_up_gate_prequant_t = ScaledGroupedMMPrequantizedRHS(
        mat_b_q=up_t_q,
        scale_b=up_s,
        mat_b_shape=tuple(module.w_up_gate.shape),
        mat_b_version=int(module.w_up_gate._version),
    )
    module._rowwise_fp8_down_prequant = ScaledGroupedMMPrequantizedRHS(
        mat_b_q=down_q,
        scale_b=down_s,
        mat_b_shape=tuple(module.w_down.shape),
        mat_b_version=int(module.w_down._version),
    )
    module._rowwise_fp8_down_prequant_t = ScaledGroupedMMPrequantizedRHS(
        mat_b_q=down_t_q,
        scale_b=down_s,
        mat_b_shape=tuple(module.w_down.transpose(1, 2).shape),
        mat_b_version=int(module.w_down._version),
    )
    module._rowwise_fp8_weight_versions = (int(module.w_up_gate._version), int(module.w_down._version))

    seen = {"calls": 0}

    def _stub_scaled_grouped_mm_q(
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        *,
        offs: torch.Tensor,
        input_grad_out: torch.Tensor | None = None,
        use_fast_accum: bool = True,
        prequantized_lhs=None,
        prequantized_rhs=None,
        prequantized_rhs_for_dgrad=None,
    ) -> torch.Tensor:
        del offs, input_grad_out, use_fast_accum, prequantized_lhs, prequantized_rhs_for_dgrad
        seen["calls"] += 1
        assert prequantized_rhs is not None
        return torch.zeros((mat_a.shape[0], mat_b.shape[-1]), dtype=mat_a.dtype, device=mat_a.device)

    def _forbid_forward_time_prequant(*args, **kwargs):
        del args, kwargs
        raise AssertionError("routed rowwise FP8 forward must not refresh/prequantize weights")

    monkeypatch.setattr(
        "olmo_core.nn.moe.v2.routed_experts.scaled_grouped_mm_q",
        _stub_scaled_grouped_mm_q,
    )
    monkeypatch.setattr(
        "olmo_core.nn.moe.v2.routed_experts.prequantize_scaled_grouped_mm_rhs",
        _forbid_forward_time_prequant,
    )

    x = torch.randn(8, 512, dtype=torch.float32)
    batch = torch.tensor([4, 4], dtype=torch.int32)
    _ = module._forward_rowwise_fp8(
        x,
        batch,
        prequantized_input_q=None,
        prequantized_input_scales=None,
    )
    assert seen["calls"] == 2


def test_swiglu_backward_grad_up_gate_compiled_helper_matches_eager(monkeypatch):
    if not torch.cuda.is_available():
        return

    monkeypatch.setenv("OLMO_MXFP8_COMPILE_SWIGLU_BWD", "1")
    torch.manual_seed(123)
    up_gate = torch.randn(16, 128, device="cuda", dtype=torch.bfloat16)
    grad_h = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16)

    actual = swiglu_backward_grad_up_gate(up_gate, grad_h)
    expected = _swiglu_backward_grad_up_gate_impl(up_gate, grad_h)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
