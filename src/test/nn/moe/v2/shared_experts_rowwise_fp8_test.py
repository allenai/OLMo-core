import types

import torch
import torch.nn.functional as F

from olmo_core.config import DType
from olmo_core.nn.moe.v2.fp8 import (
    MoERowwiseFP8Config,
    shared_experts_forward1_rowwise_fp8,
    shared_experts_forward2_rowwise_fp8,
)
from olmo_core.nn.moe.v2.shared_experts import SharedExperts
from olmo_core.nn.fp8_weight import FP8WeightStore


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
    del input_grad_out, use_fast_accum, prequantized_lhs, prequantized_rhs, prequantized_rhs_for_dgrad
    return F.grouped_mm(mat_a, mat_b, offs=offs)


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
        "olmo_core.nn.moe.v2.fp8.scaled_grouped_mm_q",
        _stub_scaled_grouped_mm_q,
    )
    monkeypatch.setattr(
        "olmo_core.nn.moe.v2.fp8.prequantize_scaled_grouped_mm_rhs",
        _forbid_forward_time_prequant,
    )

    torch.manual_seed(123)
    shared = SharedExperts(
        d_model=512,
        hidden_size=1024,
        num_experts=3,
        bias=False,
        dtype=DType.float32,
        init_device="cpu",
    )
    with torch.no_grad():
        shared.w_up_gate.normal_(mean=0.0, std=0.02)
        shared.w_down.normal_(mean=0.0, std=0.02)

    fake_self = types.SimpleNamespace(
        shared_experts=shared,
        _shared_rowwise_fp8_up_prequant=object(),
        _shared_rowwise_fp8_down_prequant=object(),
        _shared_rowwise_fp8_up_prequant_t=object(),
        _shared_rowwise_fp8_down_prequant_t=object(),
        _shared_rowwise_fp8_weight_versions=None,
        rowwise_fp8=None,
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
        "olmo_core.nn.moe.v2.fp8.scaled_grouped_mm_q",
        _stub_scaled_grouped_mm_q,
    )
    monkeypatch.setattr(
        "olmo_core.nn.moe.v2.fp8.prequantize_scaled_grouped_mm_rhs",
        _forbid_forward_time_prequant,
    )

    torch.manual_seed(123)
    shared = SharedExperts(
        d_model=512,
        hidden_size=1024,
        num_experts=3,
        bias=False,
        dtype=DType.float32,
        init_device="cpu",
    )
    with torch.no_grad():
        shared.w_up_gate.normal_(mean=0.0, std=0.02)
        shared.w_down.normal_(mean=0.0, std=0.02)

    fake_self = types.SimpleNamespace(
        shared_experts=shared,
        _shared_rowwise_fp8_up_prequant=object(),
        _shared_rowwise_fp8_down_prequant=object(),
        _shared_rowwise_fp8_up_prequant_t=object(),
        _shared_rowwise_fp8_down_prequant_t=object(),
        _shared_rowwise_fp8_weight_versions=None,
        rowwise_fp8=None,
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

    def _stub_scaled_grouped_mm_q_fp8_weight(
        mat_a: torch.Tensor,
        grad_anchor: torch.Tensor,
        *,
        offs: torch.Tensor,
        input_grad_out: torch.Tensor | None = None,
        use_fast_accum: bool = True,
        prequantized_lhs=None,
        prequantized_rhs=None,
        prequantized_rhs_for_dgrad=None,
        wgrad_sink=None,
        wgrad_sink_transpose_last2: bool = False,
        wgrad_sink_squeeze_first_dim: bool = False,
    ) -> torch.Tensor:
        del input_grad_out, use_fast_accum, prequantized_lhs, prequantized_rhs, prequantized_rhs_for_dgrad
        seen.append(
            {
                "anchor_shape": tuple(grad_anchor.shape),
                "anchor_requires_grad": grad_anchor.requires_grad,
                "sink": wgrad_sink,
                "transpose": wgrad_sink_transpose_last2,
                "squeeze": wgrad_sink_squeeze_first_dim,
            }
        )
        return F.grouped_mm(mat_a, grad_anchor, offs=offs)

    monkeypatch.setattr(
        "olmo_core.nn.moe.v2.fp8.scaled_grouped_mm_q_fp8_weight",
        _stub_scaled_grouped_mm_q_fp8_weight,
    )
    monkeypatch.setattr(
        "olmo_core.nn.moe.v2.fp8.prequantize_scaled_grouped_mm_rhs",
        _forbid_forward_time_prequant,
    )

    torch.manual_seed(123)
    shared = SharedExperts(
        d_model=512,
        hidden_size=1024,
        num_experts=3,
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
        _shared_rowwise_fp8_up_prequant=object(),
        _shared_rowwise_fp8_down_prequant=object(),
        _shared_rowwise_fp8_up_prequant_t=object(),
        _shared_rowwise_fp8_down_prequant_t=object(),
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
            "anchor_shape": (1, 512, 3 * 2 * 1024),
            "anchor_requires_grad": False,
            "sink": up_store,
            "transpose": False,
            "squeeze": True,
        },
        {
            "anchor_shape": (3, 1024, 512),
            "anchor_requires_grad": False,
            "sink": down_store,
            "transpose": False,
            "squeeze": False,
        },
    ]
