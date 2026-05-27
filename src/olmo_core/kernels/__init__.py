"""
Low-level CUDA/C++ kernels and their Python wrappers.

The wrappers are import-safe on CPU: the CUDA/C++ extensions are built lazily on first
use (see :mod:`olmo_core.kernels.cuda_extension_utils`), so importing this package does
not require a GPU or a compiler.
"""

from .grouped_mm import grouped_mm
from .mxfp8_tensor import OlmoMXFP8Tensor
from .scaled_grouped_mm import (
    ScaledGroupedMMPrequantizedLHS,
    ScaledGroupedMMPrequantizedRHS,
    prequantize_scaled_grouped_mm_rhs,
    scaled_grouped_mm_q,
    scaled_grouped_mm_q_fp8_weight,
)

__all__ = [
    "grouped_mm",
    "OlmoMXFP8Tensor",
    "ScaledGroupedMMPrequantizedLHS",
    "ScaledGroupedMMPrequantizedRHS",
    "prequantize_scaled_grouped_mm_rhs",
    "scaled_grouped_mm_q",
    "scaled_grouped_mm_q_fp8_weight",
]
