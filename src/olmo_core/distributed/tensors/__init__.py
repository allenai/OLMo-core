"""
Distributed tensor and parameter classes.
"""

from .sharded_flat_tensor import ShardedFlatTensor, ShardingSpec

__all__ = ["ShardedFlatTensor", "ShardingSpec"]
