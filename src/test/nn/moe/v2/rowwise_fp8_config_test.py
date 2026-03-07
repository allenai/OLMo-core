import pytest
import torch

from olmo_core.config import DType
from olmo_core.kernels.scaled_grouped_mm import ScaledGroupedMMPrequantizedRHS
from olmo_core.nn.moe.v2.fp8 import MoERowwiseFP8Config, normalize_rowwise_fp8_config
from olmo_core.nn.moe.v2.routed_experts import RoutedExperts


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
        d_model=64,
        hidden_size=128,
        num_experts=4,
        bias=False,
        dtype=DType.float32,
        rowwise_fp8={"enabled": True, "block_size": 32},
        init_device="cpu",
    )
    assert isinstance(module.rowwise_fp8, MoERowwiseFP8Config)
    assert module.rowwise_fp8 is not None and module.rowwise_fp8.enabled is True


def test_routed_experts_forward_rowwise_fp8_uses_cached_prequantized_rhs(monkeypatch):
    module = RoutedExperts(
        d_model=64,
        hidden_size=32,
        num_experts=2,
        bias=False,
        dtype=DType.float32,
        rowwise_fp8={"enabled": True, "block_size": 32},
        init_device="cpu",
    )

    up_q = module.w_up_gate.transpose(1, 2).to(torch.float8_e4m3fn)
    up_s = torch.ones((module.num_experts, 1), dtype=torch.float8_e8m0fnu)
    down_q = module.w_down.to(torch.float8_e4m3fn)
    down_s = torch.ones((module.num_experts, 1), dtype=torch.float8_e8m0fnu)

    module._rowwise_fp8_up_gate_prequant = ScaledGroupedMMPrequantizedRHS(
        mat_b_q=up_q,
        scale_b=up_s,
        mat_b_shape=tuple(module.w_up_gate.transpose(1, 2).shape),
        mat_b_version=int(module.w_up_gate._version),
    )
    module._rowwise_fp8_down_prequant = ScaledGroupedMMPrequantizedRHS(
        mat_b_q=down_q,
        scale_b=down_s,
        mat_b_shape=tuple(module.w_down.shape),
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
    ) -> torch.Tensor:
        del offs, input_grad_out, use_fast_accum, prequantized_lhs
        seen["calls"] += 1
        assert prequantized_rhs is not None
        return torch.zeros((mat_a.shape[0], mat_b.shape[-1]), dtype=mat_a.dtype, device=mat_a.device)

    monkeypatch.setattr(
        "olmo_core.nn.moe.v2.routed_experts.scaled_grouped_mm_q",
        _stub_scaled_grouped_mm_q,
    )

    x = torch.randn(8, 64, dtype=torch.float32)
    batch = torch.tensor([4, 4], dtype=torch.int32)
    _ = module._forward_rowwise_fp8(
        x,
        batch,
        prequantized_input_q=None,
        prequantized_input_scales=None,
    )
    assert seen["calls"] == 2
