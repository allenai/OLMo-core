"""CPU-runnable coverage for the FP8 weight-store prequantizer hook wiring.

The MXFP8/grouped-mm GEMMs themselves are GPU-only (see ``mxfp8_linear_test.py``); these tests
cover that the injectable prequantizer is wired to the right kernel for each consumer.
"""

import pytest

from olmo_core.kernels import (
    prequantize_scaled_grouped_mm_rhs,
    prequantize_scaled_mm_rhs,
)
from olmo_core.nn.fp8_weight import FP8WeightStore
from olmo_core.nn.mxfp8_linear import MXFP8Linear


def test_fp8_weight_store_defaults_to_grouped_mm_prequantizer():
    store = FP8WeightStore(logical_name="weight", logical_shape=(8, 8))
    assert store.prequantizer is prequantize_scaled_grouped_mm_rhs


def test_fp8_weight_store_accepts_injected_prequantizer():
    store = FP8WeightStore(
        logical_name="weight",
        logical_shape=(8, 8),
        prequantizer=prequantize_scaled_mm_rhs,
    )
    assert store.prequantizer is prequantize_scaled_mm_rhs


def test_mxfp8_linear_wires_scaled_mm_prequantizer():
    layer = MXFP8Linear(64, 96, bias=True)
    store = layer.fp8_weight_store
    # MXFP8Linear must inject the scaled-mm quantizer, not the grouped-mm default.
    assert store.prequantizer is prequantize_scaled_mm_rhs
    # The logical weight is exposed to the fused optimizer.
    assert [name for name, _ in layer.named_fp8_weight_stores()] == ["weight"]
    # The anchor weight is frozen (the optimizer owns the fp32 main param).
    assert layer.weight.requires_grad is False


def test_transformer_train_module_guard_rejects_unowned_fp8_stores():
    import torch.nn as nn

    from olmo_core.exceptions import OLMoConfigurationError
    from olmo_core.nn.attention import FusedAttentionV2
    from olmo_core.train.train_module.transformer.train_module import (
        _assert_no_unowned_fp8_weight_stores,
    )

    class _WithMXFP8(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = FusedAttentionV2(
                d_model=64, n_heads=4, n_kv_heads=2, head_dim=16, bias=False, mxfp8_projections=True
            )

    # MXFP8 weights are trained only by the OLMoDDP fused optimizer; the general train module's
    # ordinary optimizer would leave them frozen, so building it must fail loudly.
    with pytest.raises(OLMoConfigurationError, match="OLMoDDPTrainModule"):
        _assert_no_unowned_fp8_weight_stores(_WithMXFP8())

    # A plain model (no fp8 stores) passes.
    class _Plain(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(8, 8)

    _assert_no_unowned_fp8_weight_stores(_Plain())


def test_pipeline_per_part_fp8_guard_flags_mxfp8_part():
    # The pipeline train module runs the ownership guard on every model part before building its
    # per-part optimizers. Mirror that per-part loop: a part with MXFP8 projections must be flagged
    # (its ordinary per-part optimizer would leave the frozen anchor weights unstepped).
    import torch.nn as nn

    from olmo_core.exceptions import OLMoConfigurationError
    from olmo_core.nn.attention import FusedAttentionV2
    from olmo_core.train.train_module.transformer.train_module import (
        _assert_no_unowned_fp8_weight_stores,
    )

    class _PlainPart(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(8, 8)

    class _MXFP8Part(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = FusedAttentionV2(
                d_model=64, n_heads=4, n_kv_heads=2, head_dim=16, bias=False, mxfp8_projections=True
            )

    model_parts = [_PlainPart(), _MXFP8Part()]
    flagged = []
    for part in model_parts:
        try:
            _assert_no_unowned_fp8_weight_stores(part)
        except OLMoConfigurationError:
            flagged.append(type(part).__name__)
    assert flagged == ["_MXFP8Part"]
