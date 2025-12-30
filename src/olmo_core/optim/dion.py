import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Tuple, Type

import torch
from torch.distributed.device_mesh import DeviceMesh

from olmo_core.distributed.parallel import (
    MeshDimName,
    get_dp_model_mesh,
    get_dp_replicate_mesh,
    get_dp_shard_mesh,
    get_tp_mesh,
    get_world_mesh,
)
from olmo_core.nn.transformer import Transformer
from olmo_core.optim.config import MatrixAwareOptimConfig, OptimGroupOverride

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from dion import Dion


@dataclass
class DionConfig(MatrixAwareOptimConfig):
    """
    Configuration class for building a :class:`Dion` optimizer.

    Dion is a Muon-like optimizer that is designed to be scalable for DP-replicated, DP-shareded,
    and TP-sharded models. See https://arxiv.org/abs/2504.05295 for more details.

    Matrix-aware optimizers require a different optimizer for matrix parameters than for vector
    and embedding parameters. This optimizer is backed by AdamW for vector and embedding parameters.
    """

    lr: float = 0.01
    """Shared lr for Dion and AdamW"""

    mu: float = 0.95
    """Momentum for Dion"""

    betas: Tuple[float, float] = (0.9, 0.95)
    """Betas for AdamW"""

    weight_decay: float = 0.1
    """Weight decay for non-embedding parameters"""

    rank_fraction: float = 1.0
    """Rank fraction for Dion. Set to 1.0 for full-rank optimization."""

    @classmethod
    def optimizer(cls) -> Type["Dion"]:
        from dion import Dion  # type: ignore[reportMissingImports]

        return Dion

    def default_group_overrides(self, model: torch.nn.Module) -> list[OptimGroupOverride]:
        """
        Apply Dion's parameter grouping rules.
        """
        assert isinstance(model, Transformer)
        params = self.categorize_parameters(model)

        lm_head_out: torch.nn.Linear = model.lm_head.w_out
        model_dim = lm_head_out.weight.shape[1]

        # Matrix parameters are optimized with Dion.
        matrix_override = OptimGroupOverride(params=params["matrix"], opts=dict(algorithm="dion"))

        # Vector, embedding, and lm_head parameters are optimized with AdamW.
        embed_override = OptimGroupOverride(
            params=params["embed"], opts=dict(algorithm="adamw", weight_decay=0.0)
        )
        vector_override = OptimGroupOverride(params=params["vector"], opts=dict(algorithm="adamw"))
        lm_head_override = OptimGroupOverride(
            params=params["lm_head"],
            # lr scaled by sqrt(model_dim) for lm_head as suggested in the paper
            opts=dict(algorithm="adamw", lr=self.lr / math.sqrt(model_dim)),
        )
        return [matrix_override, vector_override, embed_override, lm_head_override]

    def build_parallelism_config(self) -> dict[str, DeviceMesh | None]:
        """
        Prepare device meshes for Dion optimizer based on the parallelism configuration.

        Supports:
        - Single-device: All meshes are None
        - DDP (not supported, would be): replicate_mesh = DP mesh, outer_shard_mesh = None
        - FSDP: outer_shard_mesh = DP mesh, replicate_mesh = None
        - HSDP: replicate_mesh = DP replicate mesh, outer_shard_mesh = DP shard mesh
        - TP: inner_shard_mesh = TP mesh (can be combined with FSDP or HSDP)

        :returns: Dictionary with 'replicate_mesh', 'outer_shard_mesh', and 'inner_shard_mesh' keys.
        """
        world_mesh = get_world_mesh()
        meshes: dict[str, DeviceMesh | None] = {
            "replicate_mesh": None,  # mesh for replicated data parallelism.
            "outer_shard_mesh": None,  # parameter sharding mesh, replicated during orthogonalization
            "inner_shard_mesh": None,  # parameter sharding mesh, remains sharded during orthogonalization
        }

        if world_mesh is None:
            return meshes
        dim_names = world_mesh.mesh_dim_names
        if dim_names is None:
            raise RuntimeError("world mesh has no dimension names")

        # Check for HSDP (has both dp_replicate and dp_shard)
        has_dp_replicate = MeshDimName.dp_replicate in dim_names
        has_dp_shard = MeshDimName.dp_shard in dim_names

        if has_dp_replicate and has_dp_shard:
            # HSDP configuration
            meshes["replicate_mesh"] = get_dp_replicate_mesh(world_mesh)
            meshes["outer_shard_mesh"] = get_dp_shard_mesh(world_mesh)
        elif MeshDimName.dp in dim_names or any(d.startswith("dp") for d in dim_names):
            # FSDP configuration
            log.warning("Cannot determine if model is FSDP or DDP, assuming FSDP.")
            meshes["outer_shard_mesh"] = get_dp_model_mesh(world_mesh)
        if MeshDimName.tp in dim_names:
            # TP configuration
            meshes["inner_shard_mesh"] = get_tp_mesh(world_mesh)
        return meshes

    def create_optimizer(self, model: torch.nn.Module, strict: bool = True, **kwargs) -> "Dion":
        """
        Create the optimizer.
        """
        from dion import Dion  # type: ignore[reportMissingImports]

        torch._dynamo.config.recompile_limit = 16

        parallelism_config = self.build_parallelism_config()
        optim: Dion = self.optimizer()(
            self.build_groups(model, strict=strict),
            replicate_mesh_grad_sync=False,  # HSDP / FSDP / DDP will handle gradient sync internally
            **parallelism_config,
            **kwargs,
        )
        return optim
