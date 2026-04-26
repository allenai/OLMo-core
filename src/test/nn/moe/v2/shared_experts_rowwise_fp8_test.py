import types

import torch
import torch.nn.functional as F

from olmo_core.config import DType
from olmo_core.nn.moe.v2.block import MoEFusedV2TransformerBlock
from olmo_core.nn.moe.v2.shared_experts import SharedExperts


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
    del input_grad_out, use_fast_accum, prequantized_lhs, prequantized_rhs
    return F.grouped_mm(mat_a, mat_b, offs=offs)


def test_shared_experts_rowwise_fp8_helpers_match_bf16_reference(monkeypatch):
    monkeypatch.setattr(
        "olmo_core.nn.moe.v2.fp8.scaled_grouped_mm_q",
        _stub_scaled_grouped_mm_q,
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
        _shared_rowwise_fp8_up_prequant=None,
        _shared_rowwise_fp8_down_prequant=None,
        _shared_rowwise_fp8_weight_versions=None,
        rowwise_fp8=None,
    )

    x = torch.randn(1, 8, 512, dtype=torch.float32)
    up, gate = MoEFusedV2TransformerBlock._shared_experts_forward1_rowwise_fp8(
        fake_self,
        x,
        use_fast_accum=True,
    )
    out_fp8 = MoEFusedV2TransformerBlock._shared_experts_forward2_rowwise_fp8(
        fake_self,
        up,
        gate,
        x.shape,
        use_fast_accum=True,
    )

    with torch.no_grad():
        out_ref = shared.forward(x)

    torch.testing.assert_close(out_fp8, out_ref)
