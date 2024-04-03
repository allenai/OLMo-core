"""
This is a light-weight, experimental rewrite of PyTorch's :class:`~torch.distributed.fsdp.FullyShardedDataParallel`
with a few of improvements, including:

- Well-defined "hands off" handling of buffers. FSDP never shards buffers, they are left as-is.
- Well-defined handling of frozen params. You can mix and match within an FSDP instance as long
  as you're consistent across the process group with which parameters are frozen.
- Full support for CPU-only training and inference via the GLOO backend.

API Reference
-------------
"""

from .fsdp import FSDP, FSDPDebugConfig, FSDPPrecision

__all__ = ["FSDP", "FSDPDebugConfig", "FSDPPrecision"]
