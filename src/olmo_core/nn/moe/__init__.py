"""
MoE layers.
"""

from .loss import MoELoadBalancingLossGranularity
from .mlp import DroplessMoEMLP, MoEMLP
from .moe import DroplessMoE, MoEBase, MoEConfig, MoEType
from .router import (
    MoELinearRouter,
    MoERouter,
    MoERouterConfig,
    MoERouterGatingFunction,
    MoERouterType,
)

def replay_routing(module, per_layer_indices):
    """Replay externally supplied selections on MoE-v2 routed-expert routers."""
    from .v2.router import replay_routing as replay_routing_v2

    return replay_routing_v2(module, per_layer_indices)


__all__ = [
    "MoEBase",
    "DroplessMoE",
    "MoEConfig",
    "MoEType",
    "MoEMLP",
    "DroplessMoEMLP",
    "MoERouter",
    "MoELinearRouter",
    "MoERouterConfig",
    "MoERouterType",
    "MoERouterGatingFunction",
    "MoELoadBalancingLossGranularity",
    "replay_routing",
]
