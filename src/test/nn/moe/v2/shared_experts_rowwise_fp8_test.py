import types

import pytest
import torch

from olmo_core.config import DType
from olmo_core.kernels import prequantize_scaled_mm_rhs
from olmo_core.nn.moe.v2.fp8 import (
    MoERowwiseFP8Config,
    shared_experts_forward1_rowwise_fp8,
    shared_experts_forward2_rowwise_fp8,
)
from olmo_core.nn.moe.v2.shared_experts import SharedExperts
from olmo_core.nn.fp8_weight import FP8WeightCacheSpec, FP8WeightStore


def _stub_scaled_mm_mxfp8_fp8_weight(
    mat_a: torch.Tensor,
    *,
    prequantized_rhs=None,
    prequantized_rhs_for_dgrad=None,
    wgrad_sink=None,
    save_wgrad_input_as_mxfp8: bool = True,
    wgrad_is_rhs: bool = False,
    wgrad_sink_unsqueeze_first_dim: bool = False,
) -> torch.Tensor:
    del prequantized_rhs_for_dgrad, wgrad_sink, save_wgrad_input_as_mxfp8, wgrad_is_rhs, wgrad_sink_unsqueeze_first_dim
    return mat_a @ prequantized_rhs


def _forbid_forward_time_prequant(*args, **kwargs):
    del args, kwargs
    raise AssertionError("shared rowwise FP8 forward must not refresh/prequantize weights")


def test_fp8_weight_store_can_accumulate_wgrad_in_fp32():
    store = FP8WeightStore(
        logical_name="w",
        logical_shape=(2, 3),
    )
    store.accumulate_wgrad_in_fp32 = True

    store.accumulate_wgrad(torch.ones(2, 3, dtype=torch.bfloat16))
    store.accumulate_wgrad(torch.ones(2, 3, dtype=torch.bfloat16))

    assert store.grad_bf16 is None
    assert store.main_grad_fp32 is not None
    assert store.main_grad_fp32.dtype == torch.float32
    torch.testing.assert_close(store.main_grad_fp32, torch.full((2, 3), 2.0))


def test_shared_experts_rowwise_fp8_helpers_match_bf16_reference(monkeypatch):
    monkeypatch.setattr(
        "olmo_core.nn.moe.v2.fp8.scaled_mm_mxfp8_fp8_weight",
        _stub_scaled_mm_mxfp8_fp8_weight,
    )
    monkeypatch.setattr(
        "olmo_core.nn.moe.v2.fp8.prequantize_scaled_mm_rhs",
        _forbid_forward_time_prequant,
    )

    torch.manual_seed(123)
    shared = SharedExperts(
        d_model=512,
        hidden_size=1024,
        num_experts=1,
        bias=False,
        dtype=DType.float32,
        init_device="cpu",
    )
    with torch.no_grad():
        shared.w_up_gate.normal_(mean=0.0, std=0.02)
        shared.w_down.normal_(mean=0.0, std=0.02)

    fake_self = types.SimpleNamespace(
        shared_experts=shared,
        _shared_rowwise_fp8_up_prequant=shared.w_up_gate.detach(),
        _shared_rowwise_fp8_down_prequant=shared.w_down.detach().squeeze(0),
        _shared_rowwise_fp8_up_prequant_t=shared.w_up_gate.detach().transpose(0, 1),
        _shared_rowwise_fp8_down_prequant_t=shared.w_down.detach().squeeze(0).transpose(0, 1),
        _shared_rowwise_fp8_weight_versions=None,
        _shared_rowwise_fp8_up_gate_weight=object(),
        _shared_rowwise_fp8_down_weight=object(),
        rowwise_fp8=MoERowwiseFP8Config(enabled=True, fp8_only_params=True),
    )

    x = torch.randn(1, 8, 512, dtype=torch.float32)
    up, gate = shared_experts_forward1_rowwise_fp8(
        fake_self,
        x,
        use_fast_accum=True,
    )
    out_fp8 = shared_experts_forward2_rowwise_fp8(
        fake_self,
        up,
        gate,
        x.shape,
        use_fast_accum=True,
    )

    with torch.no_grad():
        out_ref = shared.forward(x)

    torch.testing.assert_close(out_fp8, out_ref)


def test_shared_experts_rowwise_fp8_forward1_accepts_flattened_input(monkeypatch):
    monkeypatch.setattr(
        "olmo_core.nn.moe.v2.fp8.scaled_mm_mxfp8_fp8_weight",
        _stub_scaled_mm_mxfp8_fp8_weight,
    )
    monkeypatch.setattr(
        "olmo_core.nn.moe.v2.fp8.prequantize_scaled_mm_rhs",
        _forbid_forward_time_prequant,
    )

    torch.manual_seed(123)
    shared = SharedExperts(
        d_model=512,
        hidden_size=1024,
        num_experts=1,
        bias=False,
        dtype=DType.float32,
        init_device="cpu",
    )
    with torch.no_grad():
        shared.w_up_gate.normal_(mean=0.0, std=0.02)
        shared.w_down.normal_(mean=0.0, std=0.02)

    fake_self = types.SimpleNamespace(
        shared_experts=shared,
        _shared_rowwise_fp8_up_prequant=shared.w_up_gate.detach(),
        _shared_rowwise_fp8_down_prequant=shared.w_down.detach().squeeze(0),
        _shared_rowwise_fp8_up_prequant_t=shared.w_up_gate.detach().transpose(0, 1),
        _shared_rowwise_fp8_down_prequant_t=shared.w_down.detach().squeeze(0).transpose(0, 1),
        _shared_rowwise_fp8_weight_versions=None,
        _shared_rowwise_fp8_up_gate_weight=object(),
        _shared_rowwise_fp8_down_weight=object(),
        rowwise_fp8=MoERowwiseFP8Config(enabled=True, fp8_only_params=True),
    )

    x = torch.randn(1, 8, 512, dtype=torch.float32)
    x_flat = x.view(-1, x.shape[-1])
    up, gate = shared_experts_forward1_rowwise_fp8(
        fake_self,
        x_flat,
        use_fast_accum=True,
    )
    out_fp8 = shared_experts_forward2_rowwise_fp8(
        fake_self,
        up,
        gate,
        x.shape,
        use_fast_accum=True,
    )

    with torch.no_grad():
        out_ref = shared.forward(x)

    torch.testing.assert_close(out_fp8, out_ref)


def test_shared_experts_fp8_only_path_uses_generic_weight_stores(monkeypatch):
    seen = []

    def _stub_scaled_mm_mxfp8_fp8_weight_seen(
        mat_a: torch.Tensor,
        *,
        prequantized_rhs=None,
        prequantized_rhs_for_dgrad=None,
        wgrad_sink=None,
        save_wgrad_input_as_mxfp8: bool = True,
        wgrad_is_rhs: bool = False,
        wgrad_sink_unsqueeze_first_dim: bool = False,
    ) -> torch.Tensor:
        del prequantized_rhs_for_dgrad
        seen.append(
            {
                "rhs_shape": tuple(prequantized_rhs.shape),
                "sink": wgrad_sink,
                "save_mxfp8": save_wgrad_input_as_mxfp8,
                "rhs_wgrad": wgrad_is_rhs,
                "unsqueeze": wgrad_sink_unsqueeze_first_dim,
            }
        )
        return mat_a @ prequantized_rhs

    monkeypatch.setattr(
        "olmo_core.nn.moe.v2.fp8.scaled_mm_mxfp8_fp8_weight",
        _stub_scaled_mm_mxfp8_fp8_weight_seen,
    )
    monkeypatch.setattr(
        "olmo_core.nn.moe.v2.fp8.prequantize_scaled_mm_rhs",
        _forbid_forward_time_prequant,
    )

    torch.manual_seed(123)
    shared = SharedExperts(
        d_model=512,
        hidden_size=1024,
        num_experts=1,
        bias=False,
        dtype=DType.float32,
        init_device="cpu",
    )
    with torch.no_grad():
        shared.w_up_gate.normal_(mean=0.0, std=0.02)
        shared.w_down.normal_(mean=0.0, std=0.02)

    up_store = object()
    down_store = object()
    fake_self = types.SimpleNamespace(
        shared_experts=shared,
        _shared_rowwise_fp8_up_prequant=shared.w_up_gate.detach(),
        _shared_rowwise_fp8_down_prequant=shared.w_down.detach().squeeze(0),
        _shared_rowwise_fp8_up_prequant_t=shared.w_up_gate.detach().transpose(0, 1),
        _shared_rowwise_fp8_down_prequant_t=shared.w_down.detach().squeeze(0).transpose(0, 1),
        _shared_rowwise_fp8_weight_versions=None,
        _shared_rowwise_fp8_up_gate_weight=up_store,
        _shared_rowwise_fp8_down_weight=down_store,
        rowwise_fp8=MoERowwiseFP8Config(enabled=True, fp8_only_params=True),
    )

    x = torch.randn(1, 8, 512, dtype=torch.float32)
    up, gate = shared_experts_forward1_rowwise_fp8(
        fake_self,
        x,
        use_fast_accum=True,
    )
    out_fp8 = shared_experts_forward2_rowwise_fp8(
        fake_self,
        up,
        gate,
        x.shape,
        use_fast_accum=True,
    )

    with torch.no_grad():
        out_ref = shared.forward(x)

    torch.testing.assert_close(out_fp8, out_ref)
    assert seen == [
        {
            "rhs_shape": (512, 2 * 1024),
            "sink": up_store,
            "save_mxfp8": True,
            "rhs_wgrad": True,
            "unsqueeze": False,
        },
        {
            "rhs_shape": (1024, 512),
            "sink": down_store,
            "save_mxfp8": True,
            "rhs_wgrad": True,
            "unsqueeze": True,
        },
    ]


def test_shared_experts_rowwise_fp8_requires_single_shared_expert():
    shared = SharedExperts(
        d_model=512,
        hidden_size=1024,
        num_experts=2,
        bias=False,
        dtype=DType.float32,
        init_device="cpu",
    )
    fake_self = types.SimpleNamespace(
        shared_experts=shared,
        rowwise_fp8=MoERowwiseFP8Config(enabled=True, fp8_only_params=True),
    )
    x = torch.randn(1, 8, 512, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="exactly one shared expert"):
        shared_experts_forward1_rowwise_fp8(
            fake_self,
            x,
            use_fast_accum=True,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_shared_experts_rowwise_fp8_uses_scaled_mm_for_backward():
    torch.manual_seed(9876)
    shared = SharedExperts(
        d_model=64,
        hidden_size=64,
        num_experts=1,
        bias=False,
        dtype=DType.bfloat16,
        init_device="cuda",
    )
    with torch.no_grad():
        shared.w_up_gate.normal_(mean=0.0, std=0.02)
        shared.w_down.normal_(mean=0.0, std=0.02)

    up_store = FP8WeightStore(
        logical_name="shared_experts.w_up_gate",
        logical_shape=tuple(shared.w_up_gate.shape),
        cache_specs=(
            FP8WeightCacheSpec("rhs", lambda w: w),
            FP8WeightCacheSpec("rhs_for_dgrad", lambda w: w.transpose(0, 1)),
        ),
        optimizer_enabled=True,
        prequantizer=prequantize_scaled_mm_rhs,
    )
    down_store = FP8WeightStore(
        logical_name="shared_experts.w_down",
        logical_shape=tuple(shared.w_down.shape),
        cache_specs=(
            FP8WeightCacheSpec("rhs", lambda w: w.squeeze(0)),
            FP8WeightCacheSpec("rhs_for_dgrad", lambda w: w.squeeze(0).transpose(0, 1)),
        ),
        optimizer_enabled=True,
        prequantizer=prequantize_scaled_mm_rhs,
    )
    up_store.refresh_from_logical_weight(shared.w_up_gate)
    down_store.refresh_from_logical_weight(shared.w_down)

    fake_self = types.SimpleNamespace(
        shared_experts=shared,
        _shared_rowwise_fp8_up_prequant=up_store.require_prequantized_rhs(),
        _shared_rowwise_fp8_down_prequant=down_store.require_prequantized_rhs(),
        _shared_rowwise_fp8_up_prequant_t=up_store.require_prequantized_rhs_for_dgrad(),
        _shared_rowwise_fp8_down_prequant_t=down_store.require_prequantized_rhs_for_dgrad(),
        _shared_rowwise_fp8_weight_versions=None,
        _shared_rowwise_fp8_up_gate_weight=up_store,
        _shared_rowwise_fp8_down_weight=down_store,
        rowwise_fp8=MoERowwiseFP8Config(enabled=True, fp8_only_params=True),
    )

    x = (torch.randn(1, 32, 64, device="cuda", dtype=torch.bfloat16) * 0.05).requires_grad_()
    up, gate = shared_experts_forward1_rowwise_fp8(
        fake_self,
        x,
        use_fast_accum=True,
    )
    out = shared_experts_forward2_rowwise_fp8(
        fake_self,
        up,
        gate,
        x.shape,
        use_fast_accum=True,
    )
    out.float().sum().backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert up_store.grad_bf16 is not None
    assert up_store.grad_bf16.shape == shared.w_up_gate.shape
    assert down_store.grad_bf16 is not None
    assert down_store.grad_bf16.shape == shared.w_down.shape
