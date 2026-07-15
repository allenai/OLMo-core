"""CPU-runnable coverage for the FP8 weight-store prequantizer hook wiring.

The MXFP8/grouped-mm GEMMs themselves are GPU-only (see ``mxfp8_linear_test.py``); these tests
cover that the injectable prequantizer is wired to the right kernel for each consumer.
"""

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
