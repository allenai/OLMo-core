"""
Low-level CUDA/C++ kernels and their Python wrappers.

The wrappers are import-safe on CPU: the CUDA/C++ extensions are built lazily on first
use (see :mod:`olmo_core.kernels.cuda_extension_utils`), so importing this package does
not require a GPU or a compiler.
"""

from .grouped_mm import grouped_mm

__all__ = [
    "grouped_mm",
]
