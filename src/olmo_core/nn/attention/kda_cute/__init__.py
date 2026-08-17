"""CuTe/Triton kernels for KDA's fixed-length chunk path. See :mod:`.chunk`.

Import this package lazily — the kernel modules import the CUTLASS CuTe DSL at module
scope, which is only present in GPU environments (it rides in with flash-attn-4).
"""

from .chunk import cute_chunk_kda, cute_kda_supported

__all__ = ["cute_chunk_kda", "cute_kda_supported"]
